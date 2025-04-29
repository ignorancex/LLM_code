// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for flyinglightspeck/fls
#include "boost/date_time.hpp"
#include "boost/shared_array.hpp"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4265)
#pragma warning(disable : 4458)
#pragma warning(disable : 4100)
#pragma warning(disable : 4127)
#pragma warning(disable : 4267)
#pragma warning(disable : 4068)
#pragma warning(disable : 4245)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wshadow"
#endif //_MSC_VER
#include "ros/ros.h"
#include "flyinglightspeck/fls.h"
#include "visibility_control.h"
#include "MATLABROSMsgInterface.hpp"
#include "ROSPubSubTemplates.hpp"
class FLYINGLIGHTSPECK_EXPORT flyinglightspeck_msg_fls_common : public MATLABROSMsgInterface<flyinglightspeck::fls> {
  public:
    virtual ~flyinglightspeck_msg_fls_common(){}
    virtual void copy_from_struct(flyinglightspeck::fls* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const flyinglightspeck::fls* msg, MultiLibLoader loader, size_t size = 1);
};
  void flyinglightspeck_msg_fls_common::copy_from_struct(flyinglightspeck::fls* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //header
        const matlab::data::StructArray header_arr = arr["Header"];
        auto msgClassPtr_header = getCommonObject<std_msgs::Header>("std_msgs_msg_Header_common",loader);
        msgClassPtr_header->copy_from_struct(&msg->header,header_arr[0],loader);
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Header' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Header' is wrong type; expected a struct.");
    }
    try {
        //whatispresent
        const matlab::data::TypedArray<int8_t> whatispresent_arr = arr["Whatispresent"];
        size_t nelem = whatispresent_arr.getNumberOfElements();
        	msg->whatispresent.resize(nelem);
        	std::copy(whatispresent_arr.begin(), whatispresent_arr.begin()+nelem, msg->whatispresent.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Whatispresent' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Whatispresent' is wrong type; expected a int8.");
    }
    try {
        //coordinate
        const matlab::data::TypedArray<float> coordinate_arr = arr["Coordinate"];
        size_t nelem = coordinate_arr.getNumberOfElements();
        	msg->coordinate.resize(nelem);
        	std::copy(coordinate_arr.begin(), coordinate_arr.begin()+nelem, msg->coordinate.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Coordinate' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Coordinate' is wrong type; expected a single.");
    }
    try {
        //color
        const matlab::data::TypedArray<int8_t> color_arr = arr["Color"];
        size_t nelem = color_arr.getNumberOfElements();
        	msg->color.resize(nelem);
        	std::copy(color_arr.begin(), color_arr.begin()+nelem, msg->color.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Color' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Color' is wrong type; expected a int8.");
    }
    try {
        //duration
        const matlab::data::StructArray duration_arr = arr["Duration"];
        for (auto _durationarr : duration_arr) {
        	ros::Time _val;
        auto msgClassPtr_duration = getCommonObject<ros::Time>("ros_msg_Time_common",loader);
        msgClassPtr_duration->copy_from_struct(&_val,_durationarr,loader);
        	msg->duration.push_back(_val);
        }
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Duration' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Duration' is wrong type; expected a struct.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T flyinglightspeck_msg_fls_common::get_arr(MDFactory_T& factory, const flyinglightspeck::fls* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Header","Whatispresent","Coordinate","Color","Duration"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("flyinglightspeck/fls");
    // header
    auto currentElement_header = (msg + ctr)->header;
    auto msgClassPtr_header = getCommonObject<std_msgs::Header>("std_msgs_msg_Header_common",loader);
    outArray[ctr]["Header"] = msgClassPtr_header->get_arr(factory, &currentElement_header, loader);
    // whatispresent
    auto currentElement_whatispresent = (msg + ctr)->whatispresent;
    outArray[ctr]["Whatispresent"] = factory.createArray<flyinglightspeck::fls::_whatispresent_type::const_iterator, int8_t>({currentElement_whatispresent.size(),1}, currentElement_whatispresent.begin(), currentElement_whatispresent.end());
    // coordinate
    auto currentElement_coordinate = (msg + ctr)->coordinate;
    outArray[ctr]["Coordinate"] = factory.createArray<flyinglightspeck::fls::_coordinate_type::const_iterator, float>({currentElement_coordinate.size(),1}, currentElement_coordinate.begin(), currentElement_coordinate.end());
    // color
    auto currentElement_color = (msg + ctr)->color;
    outArray[ctr]["Color"] = factory.createArray<flyinglightspeck::fls::_color_type::const_iterator, int8_t>({currentElement_color.size(),1}, currentElement_color.begin(), currentElement_color.end());
    // duration
    auto currentElement_duration = (msg + ctr)->duration;
    auto msgClassPtr_duration = getCommonObject<ros::Time>("ros_msg_Time_common",loader);
    outArray[ctr]["Duration"] = msgClassPtr_duration->get_arr(factory,&currentElement_duration[0],loader,currentElement_duration.size());
    }
    return std::move(outArray);
  } 
class FLYINGLIGHTSPECK_EXPORT flyinglightspeck_fls_message : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~flyinglightspeck_fls_message(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
};  
  std::shared_ptr<MATLABPublisherInterface> 
          flyinglightspeck_fls_message::generatePublisherInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSPublisherImpl<flyinglightspeck::fls,flyinglightspeck_msg_fls_common>>();
  }
  std::shared_ptr<MATLABSubscriberInterface> 
         flyinglightspeck_fls_message::generateSubscriberInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSSubscriberImpl<flyinglightspeck::fls,flyinglightspeck::fls::ConstPtr,flyinglightspeck_msg_fls_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface>
         flyinglightspeck_fls_message::generateRosbagWriterInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSBagWriterImpl<flyinglightspeck::fls,flyinglightspeck_msg_fls_common>>();
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(flyinglightspeck_msg_fls_common, MATLABROSMsgInterface<flyinglightspeck::fls>)
CLASS_LOADER_REGISTER_CLASS(flyinglightspeck_fls_message, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1