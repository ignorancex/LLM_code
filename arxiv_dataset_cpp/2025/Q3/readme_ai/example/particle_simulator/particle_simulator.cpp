#include <SFML/Graphics.hpp>
#include <hedgehog/hedgehog.h>
#include <random>
#include <vector>

// Define a simple particle structure
struct Particle {
    float x, y;
    float vx, vy;
};

// Define a block structure to manage particles within a spatial area
class Block {
public:
    // Default constructor
    Block() : xMin(0.0f), xMax(0.0f), yMin(0.0f), yMax(0.0f), haloWidth(0.0f) {}

    // Constructor to initialize block boundaries
    Block(float xMin, float xMax, float yMin, float yMax, float haloWidth)
        : xMin(xMin), xMax(xMax), yMin(yMin), yMax(yMax), haloWidth(haloWidth) {}

    // Method to check if a particle is within the block's boundaries
    bool contains(const Particle& particle) const {
        return particle.x >= xMin && particle.x < xMax && particle.y >= yMin && particle.y < yMax;
    }

    // Method to add a particle to the block
    void addParticle(const Particle& particle) {
        particles.push_back(particle);
    }

    // Method to update the halo region
    void updateHalo(const std::vector<Block>& neighboringBlocks, const std::vector<Particle>& particles) {
        halo.clear();
        for (const auto& particle : particles) {
            if (particle.x >= xMin - haloWidth && particle.x < xMax + haloWidth &&
                particle.y >= yMin - haloWidth && particle.y < yMax + haloWidth) {
                halo.push_back(particle);
            }
        }
    }

    std::vector<Particle> particles;
    float xMin, xMax, yMin, yMax;
    float haloWidth;
    std::vector<Particle> halo;
};

// Define a task to simulate particle movement and collision within blocks
class ParticleSimulationTask : public hh::AbstractTask<1, Block, Block> {
public:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis;
    
    // Constructor to initialize the task with a random number generator and distribution
    ParticleSimulationTask(size_t numberThreads, std::mt19937 gen, std::uniform_real_distribution<float> dis)
        : hh::AbstractTask<1, Block, Block>("Particle Simulation Task", numberThreads),
          gen(gen), dis(dis) {}
    
    // Execute method to simulate particle movement and collision
    void execute(std::shared_ptr<Block> block, int width, int height) {
        // Update particle positions based on the fixed timestep dt
        for (auto& particle : block->particles) {
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Ensure particle stays within boundaries
            particle.x = std::max(0.0f, std::min(particle.x, static_cast<float>(width - 1)));
            particle.y = std::max(0.0f, std::min(particle.y, static_cast<float>(height - 1)));

            // Boundary collision detection and response within the global simulation area
            if (particle.x <= 0 || particle.x >= width - 1) {
                particle.vx = -particle.vx;
            }
            if (particle.y <= 0 || particle.y >= height - 1) {
                particle.vy = -particle.vy;
            }

            // Ensure particles are always moving
            if (particle.vx == 0 && particle.vy == 0) {
                particle.vx = (dis(gen) - 0.5f) * 0.1f;
                particle.vy = (dis(gen) - 0.5f) * 0.1f;
            }
        }

        for (auto& particle : block->particles) {
            float minDistance = 10.0f;
            float dx, dy, distance, nx, ny, tx, ty;
            float v1n, v1t, v2n, v2t, v1nAfter, v2nAfter;

            // Check for collisions with halo particles and update their velocities
            for (auto& other : block->halo) {
                dx = particle.x - other.x;
                dy = particle.y - other.y;
                distance = std::sqrt(dx * dx + dy * dy);
             
                if (distance < minDistance) {
                    nx = dx / distance;
                    ny = dy / distance;
                    tx = -ny;
                    ty = nx;
             
                    v1n = particle.vx * nx + particle.vy * ny;
                    v1t = particle.vx * tx + particle.vy * ty;
                    v2n = other.vx * nx + other.vy * ny;
                    v2t = other.vx * tx + other.vy * ty;
             
                    v1nAfter = v2n;
                    v2nAfter = v1n;
             
                    particle.vx = v1nAfter * nx + v1t * tx;
                    particle.vy = v1nAfter * ny + v1t * ty;
             
                    other.vx = v2nAfter * nx + v2t * tx;
                    other.vy = v2nAfter * ny + v2t * ty;
             
                    float overlap = minDistance - distance;
                    particle.x += nx * overlap / 2;
                    particle.y += ny * overlap / 2;
                }
            }

            // Check for collisions with other particles in the block
            for (auto& other : block->particles) {
                if (&particle != &other) {
                    dx = particle.x - other.x;
                    dy = particle.y - other.y;
                    distance = std::sqrt(dx * dx + dy * dy);
                    if (distance < minDistance) {
                        nx = dx / distance;
                        ny = dy / distance;
                        tx = -ny;
                        ty = nx;

                        v1n = particle.vx * nx + particle.vy * ny;
                        v1t = particle.vx * tx + particle.vy * ty;
                        v2n = other.vx * nx + other.vy * ny;
                        v2t = other.vx * tx + other.vy * ty;

                        v1nAfter = v2n;
                        v2nAfter = v1n;

                        particle.vx = v1nAfter * nx + v1t * tx;
                        particle.vy = v1nAfter * ny + v1t * ty;
                        other.vx = v2nAfter * nx + v2t * tx;
                        other.vy = v2nAfter * ny + v2t * ty;

                        float overlap = minDistance - distance;
                        particle.x += nx * overlap / 2;
                        particle.y += ny * overlap / 2;
                        other.x -= nx * overlap / 2;
                        other.y -= ny * overlap / 2;
                    }
                }
            }
        }
        this->addResult(block);
    }

    // Overloaded execute method to allow for default width and height
    void execute(std::shared_ptr<Block> block) {
        execute(block, 800, 600); // Default width and height
    }

    // Copy method to create a new instance of the task with the same number of threads
    std::shared_ptr<hh::AbstractTask<1, Block, Block>> copy() {
        return std::make_shared<ParticleSimulationTask>(this->numberThreads(), gen, dis);
    }
};

// Define a state to manage the simulation state and directly pass blocks
class ParticleDistributionState : public hh::AbstractState<1, std::vector<Particle>, Block> {
public:
    void execute(std::shared_ptr<std::vector<Particle>> data) override {
        particles_ = *data;

        const int blockSize = 200;
        const int width = 800;
        const int height = 600;
        const float haloWidth = 10.0f;
        int numBlocksX = width / blockSize;
        int numBlocksY = height / blockSize;

        // First, create all blocks and load particles into them
        std::vector<Block> blocks(numBlocksX * numBlocksY);
        for (int blockIndex = 0; blockIndex < numBlocksX * numBlocksY; ++blockIndex) {
            int i = blockIndex / numBlocksY;
            int j = blockIndex % numBlocksY;
            float xMin = i * blockSize;
            float xMax = (i + 1) * blockSize;
            float yMin = j * blockSize;
            float yMax = (j + 1) * blockSize;
            blocks[blockIndex] = Block(xMin, xMax, yMin, yMax, haloWidth);

            for (auto& particle : particles_) {
                if (blocks[blockIndex].contains(particle)) {
                    blocks[blockIndex].addParticle(particle);
                }
            }
        }

        // Then, update the halo region for each block
        for (int blockIndex = 0; blockIndex < numBlocksX * numBlocksY; ++blockIndex) {
            int i = blockIndex / numBlocksY;
            int j = blockIndex % numBlocksY;
            Block& block = blocks[blockIndex];

            std::vector<Block> neighboringBlocks;

            // Collect neighboring blocks within a 3x3 grid
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    int neighborIndex = (i + x) * numBlocksY + (j + y);
                    if (x == 0 && y == 0) continue;
                    if (neighborIndex >= 0 && neighborIndex < numBlocksX * numBlocksY) {
                        neighboringBlocks.push_back(blocks[neighborIndex]);
                    }
                }
            }

            // Update the halo region of the block with particles from neighboring blocks
            for (const auto& neighbor : neighboringBlocks) {
                for (const auto& particle : neighbor.particles) {
                    if (particle.x >= block.xMin - block.haloWidth && particle.x < block.xMax + block.haloWidth &&
                        particle.y >= block.yMin - block.haloWidth && particle.y < block.yMax + block.haloWidth) {
                        block.halo.push_back(particle);
                    }
                }
            }

            this->addResult(std::make_shared<Block>(block));
        }
    }

private:
    std::vector<Particle> particles_;
};

// Main function
int main() {
    // Initialize SFML window
    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(800, 600), 32), "Particle Simulator");

    // Initialize particles
    std::random_device rd;
    std::mt19937 gen(rd());
    const int width = 800;
    const int height = 600;
    std::uniform_real_distribution<float> dis(0.0f, width);
    std::uniform_real_distribution<float> velDis(-0.75f, 0.75f);
    std::shared_ptr<std::vector<Particle>> particles = std::make_shared<std::vector<Particle>>(100);

    // Initialize particles with random positions and velocities
    for (auto& particle : *particles) {
        particle.x = dis(gen);
        particle.y = dis(gen);
        particle.vx = velDis(gen);
        particle.vy = velDis(gen);
    }

    // Create a Hedgehog graph
    auto graph = std::make_shared<hh::Graph<1, std::vector<Particle>, Block>>();

    // Create tasks and state managers
    auto distributionStateManager = std::make_shared<hh::StateManager<1, std::vector<Particle>, Block>>(std::make_shared<ParticleDistributionState>());
    auto simulationTask = std::make_shared<ParticleSimulationTask>(std::thread::hardware_concurrency(), gen, dis);

    // Configure the graph
    graph->inputs(distributionStateManager);
    graph->edges(distributionStateManager, simulationTask);
    graph->output<Block>(simulationTask);

    // Execute the graph
    graph->executeGraph();

    // Push initial particles to the graph
    graph->pushData(particles);

    std::vector<Block> latestBlocks;

    // Main loop
    while (window.isOpen()) {
        if (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>() || event->is<sf::Event::KeyPressed>()) {
                window.close();
            }
        }

        latestBlocks.clear();
        
        // Collect results for each block
        for (int blockIndex = 0; blockIndex < (width / 200) * (height / 200); ++blockIndex) {
            if (auto result = graph->getBlockingResult()) {
                latestBlocks.push_back(*std::get<std::shared_ptr<Block>>(*result));
            }
        }

        window.clear();

        const float radius = 5.0f;
        const int numPoints = 10; // Number of points to approximate a circle
        sf::VertexArray circles(sf::PrimitiveType::Triangles, particles->size() * numPoints * 3);
        for (size_t i = 0; i < particles->size(); ++i) {
            const auto& particle = (*particles)[i];
            float x = particle.x;
            float y = particle.y;

            sf::Vector2f center(x, y);
            for (int j = 0; j < numPoints; ++j) {
                float angle1 = j * 2 * 3.14159f / numPoints;
                float angle2 = (j + 1) * 2 * 3.14159f / numPoints;

                sf::Vector2f vertex1(x + radius * std::cos(angle1), y + radius * std::sin(angle1));
                sf::Vector2f vertex2(x + radius * std::cos(angle2), y + radius * std::sin(angle2));

                circles[(i * numPoints + j) * 3].position = center;
                circles[(i * numPoints + j) * 3].color = sf::Color::Red;
                circles[(i * numPoints + j) * 3 + 1].position = vertex1;
                circles[(i * numPoints + j) * 3 + 1].color = sf::Color::Red;
                circles[(i * numPoints + j) * 3 + 2].position = vertex2;
                circles[(i * numPoints + j) * 3 + 2].color = sf::Color::Red;
            }
        }
        window.draw(circles);

        window.display();

        // Update particles based on the latest simulation results
        particles->clear();
        for (const auto& block : latestBlocks) {
            for (const auto& particle : block.particles) {
                particles->push_back(particle);
            }
        }

        graph->pushData(particles);
    }

    // Indicate that no more data will be sent
    graph->finishPushingData();
    graph->waitForTermination();
    return 0;
}

