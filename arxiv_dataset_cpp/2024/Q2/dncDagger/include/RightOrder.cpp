#include "RightOrder.h"
#include "LeftOrder.h"
using namespace std;

RightOrder::RightOrder(vector<int> &order,
                       double order_score,
                       vector<double> &node_scores,
                       size_t n) : order(order), // O(p)
                                   order_score(order_score), // O(1)
                                   node_scores(node_scores), // O(p)
                                   n_nodes(n) // O(1)
{
  inserted_max_order_scores = vector<double>(order.size()); // O(p)
  new_top_scores = vector<double>(order.size(), 0); // O(p)
  best_insert_pos = vector<size_t>(order.size());
  upper_bound = 1.0; // set by edmond
  upper_bound_hidden = 1.0; // set by edmond
  hidden_top_score_sum = 0.0; // set to correct value on init
  
}

int RightOrder::front() const
{
  return order[order.size() - n_nodes];
}

size_t RightOrder::front_ind() const
{
  return order.size() - n_nodes;
}

size_t RightOrder::size() const
{
  return n_nodes;
}

size_t RightOrder::size_hidden() const
{
  return order.size() - n_nodes;
}

vector<int>::reverse_iterator RightOrder::begin()
{
  return order.rbegin();
}

vector<int>::reverse_iterator RightOrder::end()
{
  return order.rbegin() + n_nodes;
}

vector<int>::iterator RightOrder::rbegin()
{
  return order.end() - n_nodes;
}
vector<int>::iterator RightOrder::rend()
{
  return order.end();
}

vector<int>::iterator RightOrder::hidden_begin()
{
  return order.begin() ;
}

vector<int>::iterator RightOrder::hidden_end()
{
  return order.begin() + order.size() - n_nodes;
}

RightOrder operator+(RightOrder &ro, LeftOrder &lo)
{
  vector<int> order;
  order.reserve(ro.n_nodes + lo.n);
  order.insert(order.end(), lo.begin(), lo.end());
  order.insert(order.end(), ro.rbegin(), ro.rend());

  double order_score = ro.order_score + lo.order_score;
  // This has to be fixe as well
  size_t n = ro.n_nodes + lo.n;

  vector<double> node_scores(n, -1);
  // loop trought all the nodes.
  for (auto i = lo.begin(); i != lo.end(); i++)
  {
    node_scores[*i] = lo.node_scores[*i];
  }
  for (auto i = ro.begin(); i != ro.end(); i++)
  {
    node_scores[*i] = ro.node_scores[*i];
  }

  RightOrder c(order, order_score, node_scores, n);
  return c;
}

RightOrder operator+(LeftOrder &lo, RightOrder &ro)
{
  return ro + lo;
}

ostream &operator<<(ostream &os, const RightOrder &ro)
{
  size_t p = ro.order.size();

  if (ro.n_nodes != p)
  {
    os << "([...],";
  }
  else
  {
    os << "(";
  }

  for (size_t i = p - ro.n_nodes; i < p; i++)
  {
    if (i != p - 1)
    {
      os << ro.order[i] << ", ";
    }
    else
    {
      os << ro.order[i];
    }
  }

  os << ")";
  os << ": " << ro.order_score;
  return os;
}