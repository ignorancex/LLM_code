#include <path_searching/astar.h>
#include <sstream>

using namespace std;
using namespace Eigen;

namespace GraphSearch {
Astar::~Astar() {
  for (int i = 0; i < allocate_num_; i++) {
    delete path_node_pool_[i];
  }
}

int Astar::search(Eigen::Vector2d start_pt, Eigen::Vector2d end_pt) {
  /* ---------- initialize ---------- */
  NodePtr cur_node = path_node_pool_[0];
  cur_node->parent = NULL;
  cur_node->index = map_util_->floatToInt(start_pt);
  cur_node->position = map_util_->intToFloat(cur_node->index);
  std::cout << "begin to astar search" << std::endl;
  cur_node->g_score = 0.0;

  Eigen::Vector2d end_state(2);
  Eigen::Vector2i end_index;

  end_index = map_util_->floatToInt(end_pt);
  end_pt = map_util_->intToFloat(end_index);
  cur_node->f_score = lambda_heu_ * getEuclHeu(cur_node->position, end_pt);
  cur_node->node_state = IN_OPEN_SET;

  open_set_.push(cur_node);
  use_node_num_ += 1;

  expanded_nodes_.insert(cur_node->index, cur_node);

  NodePtr neighbor = NULL;
  NodePtr terminate_node = NULL;

  /* ---------- search loop ---------- */
  while (!open_set_.empty()) {
    /* ---------- get lowest f_score node ---------- */
    cur_node = open_set_.top();

    /* ---------- determine termination ---------- */

    bool reach_end = abs(cur_node->index(0) - end_index(0)) <= 1 &&
        abs(cur_node->index(1) - end_index(1)) <= 1;

    if (reach_end) {
      cout << "[Astar]:---------------------- " << use_node_num_ << endl;
      cout << "use node num: " << use_node_num_ << endl;
      cout << "iter num: " << iter_num_ << endl;
      terminate_node = cur_node;
      retrievePath(terminate_node);
      has_path_ = true;

      return REACH_END;
    }

    /* ---------- pop node and add to close set ---------- */
    open_set_.pop();
    cur_node->node_state = IN_CLOSE_SET;
    iter_num_ += 1;

    /* ---------- init neighbor expansion ---------- */
    Eigen::Vector2d cur_pos = cur_node->position;
    Eigen::Vector2d pro_pos;
    double pro_t;

    vector<Eigen::Vector2d> inputs;
    Eigen::Vector2d d_pos;

    /* ---------- expansion loop ---------- */
    for (int didx_x = -1; didx_x <= 1; didx_x ++)
      for (double didx_y = -1; didx_y <= 1; didx_y ++){
        d_pos << resolution_ * didx_x, resolution_ * didx_y;
        if (d_pos.norm() < 1e-3) continue;
        pro_pos = cur_pos + d_pos;
        Eigen::Vector2i pro_id = map_util_->floatToInt(pro_pos);
        /* ---------- check if in feasible space ---------- */
        /* inside map range */
        if (map_util_->isOutside(pro_id)) {
          continue;
        }
        /* not in close set */
        NodePtr pro_node = expanded_nodes_.find(pro_id);

        if (pro_node != NULL && pro_node->node_state == IN_CLOSE_SET) {
          continue;
        }
        /* collision free */
        bool isOccupied = false;
        if(map_util_->getDistance(pro_pos) < 0.35){
            isOccupied = true;
        }
        if(isOccupied){
          continue;
        }
        /* ---------- compute cost ---------- */
        double tmp_g_score, tmp_f_score;
        tmp_g_score = d_pos.norm() + cur_node->g_score;
        tmp_f_score = tmp_g_score + lambda_heu_ * getEuclHeu(pro_pos, end_pt);
        if (pro_node == NULL) {
          pro_node = path_node_pool_[use_node_num_];
          pro_node->index = pro_id;
          pro_node->position = pro_pos;
          pro_node->f_score = tmp_f_score;
          pro_node->g_score = tmp_g_score;
          pro_node->parent = cur_node;
          pro_node->node_state = IN_OPEN_SET;
          open_set_.push(pro_node);
          expanded_nodes_.insert(pro_id, pro_node);
          use_node_num_ += 1;
          if (use_node_num_ == allocate_num_) {
            cout << "run out of memory." << endl;
            return NO_PATH;
          }
        } else if (pro_node->node_state == IN_OPEN_SET) {
          if (tmp_g_score < pro_node->g_score) {
            pro_node->position = pro_pos;
            pro_node->f_score = tmp_f_score;
            pro_node->g_score = tmp_g_score;
            pro_node->parent = cur_node;
          }
        } else {
          cout << "error type in searching: " << pro_node->node_state << endl;
        }

        /* ----------  ---------- */
      }
  }

  /* ---------- open set empty, no path ---------- */
  cout << "open set empty, no path!" << endl;
  cout << "use node num: " << use_node_num_ << endl;
  cout << "iter num: " << iter_num_ << endl;
  return NO_PATH;
}

void Astar::setParam(ros::NodeHandle& nh) {
  nh.param("astar/resolution_astar", resolution_, 0.1);
  nh.param("astar/time_resolution", time_resolution_, 1.0);
  nh.param("astar/lambda_heu", lambda_heu_, 1.0);
  nh.param("astar/allocate_num", allocate_num_, 100000);
  tie_breaker_ = 1.0 + 1.0 / 10000;

}

void Astar::retrievePath(NodePtr end_node) {
  NodePtr cur_node = end_node;
  path_nodes_.push_back(cur_node);

  while (cur_node->parent != NULL) {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node);
  }

  reverse(path_nodes_.begin(), path_nodes_.end());
}

std::vector<Eigen::Vector2d> Astar::getPath() {
  vector<Eigen::Vector2d> path;
  for (int i = 0; i < path_nodes_.size(); ++i) {
    path.push_back(path_nodes_[i]->position);
  }
  return path;
}

double Astar::getDiagHeu(Eigen::Vector2d x1, Eigen::Vector2d x2) {
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));

  double h;
  int diag = min(dx, dy);
  dx -= diag;
  dy -= diag;

  if (dx < 1e-4) {
    h = 1.0 * sqrt(2.0) * diag + 1.0 * abs(dy);
  }
  if (dy < 1e-4) {
    h = 1.0 * sqrt(2.0) * diag  + 1.0 * abs(dx);
  }

  return tie_breaker_ * h;
}

double Astar::getManhHeu(Eigen::Vector2d x1, Eigen::Vector2d x2) {
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));

  return tie_breaker_ * (dx + dy);
}

double Astar::getEuclHeu(Eigen::Vector2d x1, Eigen::Vector2d x2) {
  // return 0.0;
  return tie_breaker_ * (x2 - x1).norm();
}

void Astar::init() {
  /* ---------- map params ---------- */
  this->inv_resolution_ = 1.0 / resolution_;
  inv_time_resolution_ = 1.0 / time_resolution_;
  /* ---------- pre-allocated node ---------- */
  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++) {
    path_node_pool_[i] = new Node;
  }

  use_node_num_ = 0;
  iter_num_ = 0;
}

void Astar::setEnvironment(map_util::OccMapUtil* env) {
  map_util_ = env;
}

void Astar::reset() {
  expanded_nodes_.clear();
  path_nodes_.clear();

  std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator0> empty_queue;
  open_set_.swap(empty_queue);

  for (int i = 0; i < use_node_num_; i++) {
    NodePtr node = path_node_pool_[i];
    node->parent = NULL;
    node->node_state = NOT_EXPAND;
  }

  use_node_num_ = 0;
  iter_num_ = 0;
}

std::vector<NodePtr> Astar::getVisitedNodes() {
  vector<NodePtr> visited;
  visited.assign(path_node_pool_.begin(), path_node_pool_.begin() + use_node_num_ - 1);
  return visited;
}
}  // namespace rm_planner
