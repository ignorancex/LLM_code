
#include "LeftOrder.h"

LeftOrder::LeftOrder(vector<int> &order,
                     double order_score,
                     vector<double> &node_scores,
                     size_t n) : order(order),
                                 order_score(order_score),
                                 node_scores(node_scores),
                                 n(n)
{
  inserted_max_order_scores = vector<double>(order.size());
  new_back_scores = vector<double>(order.size());
  best_insert_pos = vector<size_t>(order.size());
}

int LeftOrder::back() const
{
  return order[n - 1];
}

size_t LeftOrder::back_ind() const
{
  return n - 1;
}

vector<int>::iterator LeftOrder::begin(){
  return order.begin();
}

vector<int>::iterator LeftOrder::end(){
  return order.begin() + n ;
}

vector<int>::iterator LeftOrder::hidden_begin(){
  return order.begin();
}

vector<int>::iterator LeftOrder::hidden_end(){
  return order.begin() + n;
}

ostream &operator<<(ostream &os, const LeftOrder &lo)
{
  size_t p = lo.order.size();


  os << "(";

//  iterator it = lo.begin();
  // for (vector<int>::iterator it = lo.begin(); it != lo.end(); it++)
  // {
  //   if (it != lo.end() - 1)
  //   {
  //     os << *it << ", ";
  //   }
  //   else
  //   {
  //     os << *it;
  //   }
  // }

  for (size_t i = 0; i < lo.n;  i++)
  {
    if (i != lo.n-1)
    {
      os << lo.order[i] << ", ";
    }
    else
    {
      os << lo.order[i];
    }
  }
  if (lo.n != lo.n-1)
  {
    os << ",[...])";
  }
  else
  {
    os << ")";
  }
  os << ": " << lo.order_score;

  return os;
}