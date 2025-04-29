#include "rclcpp/rclcpp.hpp"
#include "custom_interfaces/msg/wheel_encoders.hpp"
#include <boost/asio.hpp>
#include <string>
#include <chrono>

class EncoderNode : public rclcpp::Node
{
public:
    EncoderNode()
        : Node("encoder_node"), port_LW_(io_service_, "/dev/mc_LW"), port_RW_(io_service_, "/dev/mc_RW")
    {
        publisher_ = this->create_publisher<custom_interfaces::msg::WheelEncoders>("wheel/encoders", 10);
        initialize_port(port_LW_);
        initialize_port(port_RW_);

        RCLCPP_INFO(this->get_logger(), "Serial ports initialized.");

        std::thread([this]() { io_service_.run(); }).detach();
    }

private:
    struct EncoderPort
    {
        boost::asio::serial_port port;
        boost::asio::streambuf buffer;
        std::chrono::steady_clock::time_point last_received;
        std::string port_name;
        rclcpp::TimerBase::SharedPtr timer;

        EncoderPort(boost::asio::io_service& io_service, const std::string& name)
            : port(io_service), port_name(name)
        {}
    };

    rclcpp::Publisher<custom_interfaces::msg::WheelEncoders>::SharedPtr publisher_;
    boost::asio::io_service io_service_;
    EncoderPort port_LW_, port_RW_;
    const std::chrono::seconds timeout_duration_ = std::chrono::seconds(1);
    double left_value = 0.0;
    double right_value = 0.0;

    void initialize_port(EncoderPort& encoder_port)
    {
        encoder_port.port.open(encoder_port.port_name);
        encoder_port.port.set_option(boost::asio::serial_port_base::baud_rate(9600));
        setup_async_read(encoder_port);

        encoder_port.timer = this->create_wall_timer(
            timeout_duration_,
            [this, &encoder_port]() { check_timeout_and_reconnect(encoder_port); }
        );
    }


    void check_timeout_and_reconnect(EncoderPort& encoder_port)
    {
        auto now = std::chrono::steady_clock::now();
        if (now - encoder_port.last_received > timeout_duration_)
        {
            RCLCPP_WARN(this->get_logger(), 
                        "%s timeout detected! Attempting to reconnect...", 
                        encoder_port.port_name.c_str());

            while (true)
            {
                try
                {
                    if(encoder_port.port.is_open())
                    {
                        encoder_port.port.close();
                    }

                    encoder_port.port.open(encoder_port.port_name);
                    encoder_port.port.set_option(boost::asio::serial_port_base::baud_rate(9600));
                    encoder_port.last_received = std::chrono::steady_clock::now();

                    RCLCPP_INFO(this->get_logger(), 
                                "Successfully reconnected to %s.", 
                                encoder_port.port_name.c_str());

                    setup_async_read(encoder_port);
                    break;
                }
                catch (const boost::system::system_error& e)
                {
                    RCLCPP_WARN(this->get_logger(), 
                                "Failed to reconnect to %s. Retrying in 0.5 seconds...", 
                                encoder_port.port_name.c_str());

                    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }
    }

    void setup_async_read(EncoderPort& encoder_port)
    {
        boost::asio::async_read_until(encoder_port.port, encoder_port.buffer, "\n",
            [this, &encoder_port](boost::system::error_code ec, std::size_t bytes_transferred)
            {
                if (!ec)
                {
                    std::istream is(&encoder_port.buffer);
                    std::string line;
                    std::getline(is, line);

                    encoder_port.last_received = std::chrono::steady_clock::now();
                    process_received_data(encoder_port, line);

                    encoder_port.buffer.consume(bytes_transferred);
                    setup_async_read(encoder_port);
                }
                else
                {
                    RCLCPP_ERROR(this->get_logger(), 
                                 "Error reading from %s: %s", 
                                 encoder_port.port_name.c_str(), 
                                 ec.message().c_str());
                }
            });
    }

    void process_received_data(EncoderPort& encoder_port, const std::string& line)
    {
        if(encoder_port.port_name == "/dev/mc_LW")
        {
            left_value = std::stod(line);
        }
        else if(encoder_port.port_name == "/dev/mc_RW")
        {
            right_value = std::stod(line);
        }

        custom_interfaces::msg::WheelEncoders msg;
        msg.header.stamp = this->now();
        msg.left_encoder = left_value;
        msg.right_encoder = right_value;

        publisher_->publish(msg);
        // RCLCPP_INFO(this->get_logger(), "Published: Left Encoder: %f, Right Encoder: %f", left_value, right_value);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EncoderNode>());
    rclcpp::shutdown();
    return 0;
}
