/////////////////////////////////////////////////////////////////////////////
//
//                          ROSIA
//
// This package contains the source code which implements the
// rotation-search-based star identification algorithm proposed in
//
// @article{chng2022rosia,
//  title={ROSIA: Rotation-Search-Based Star Identification Algorithm},
//  author={Chng, Chee-Kheng and Bustos, Alvaro Parra and McCarthy, Benjamin and Chin, Tat-Jun},
//  journal={arXiv preprint arXiv:2210.00429},
//  year={2022}
//}
//
// Copyright (c) 2023 Chee-Kheng Chng
// Australian Institute for Machine Learning (AIML)
// School of Computer Science, The University of Adelaide, Australia
// Please acknowledge the authors by citing the above paper in any academic
// publications that have made use of this package or part of it.
//
/////////////////////////////////////////////////////////////////////////////


#include "reg_binary_tree.h"
//#include "mex.h"

using namespace reg::binarytree;


treeNode *reg::binarytree::Insert(treeNode *node,payload newdata,treeNode *caller)
{
    if(node==NULL)
    {
        treeNode *temp;
        temp = (treeNode *)malloc(sizeof(treeNode));
        temp->data = newdata;
        temp->left = temp -> right = NULL;
        return temp;
    }
    if(newdata.val>=(node->data.val))
    {
        node->right = Insert(node->right, newdata, node);
    }
    else if(newdata.val<(node->data.val))
    {
        node->left = Insert(node->left, newdata, node);
    }
    return node;
}

void reg::binarytree::free_Binarytree(treeNode *node)
{
    if(node==NULL)
    {
        return;
    }
    free_Binarytree(node->left);
    free_Binarytree(node->right);
    free(node);
}

int reg::binarytree::queryLower(treeNode *node, double qval, treeNode *caller)
{
    if(node==NULL)
    {
        if (qval<=caller->data.val)
        {
            return caller->data.order;
        }
        else
        {
            return caller->data.order+1;
        }
    }
    if(qval>node->data.val)
    {
        return queryLower(node->right, qval, node);
    }
    return queryLower(node->left, qval, node);

}

int reg::binarytree::queryUpper(treeNode *node,double qval,treeNode *caller)
{
    if(node==NULL)
    {
        if (qval>=caller->data.val)
        {
            return caller->data.order;
        }
        else
        {
            return caller->data.order-1;
        }
    }
    if(qval>=node->data.val)
    {
        return queryUpper(node->right,qval,node);
    }

    return queryUpper(node->left,qval,node);
}

double reg::binarytree::queryMiddle(treeNode *node,double qval,treeNode *caller)
{
    if(node==NULL)
    {
        if (qval<caller->data.val)
        {
            return (double)(caller->data.order)-0.5;
        }
        else
        {
            return (double)(caller->data.order)+0.5;
        }
    }
    if(qval>node->data.val)
    {
        return queryMiddle(node->right,qval,node);
    }
    else if(qval<node->data.val)
    {
        return queryMiddle(node->left,qval,node);
    }
    else
    {
        if (fmod((double)(node->data.order),2.0)==1.0)
        {
            return (double)(node->data.order)-0.5;
        }
        else
        {
            return (double)(node->data.order)+0.5;
        }
    }
}

void reg::binarytree::PrintInorder(treeNode *node)
{
    if(node==NULL)
    {        
        return;
    }
    PrintInorder(node->left);
    printf("%d %f ",node->data.order,node->data.val);
    PrintInorder(node->right);
}


int reg::binarytree::size_Binarytree(treeNode *node)
{
    if (node==NULL)
    {
        return 0;
    }
    return size_Binarytree(node->left)+size_Binarytree(node->right)+1;
}


int reg::binarytree::count_pointers_Binarytree(treeNode *node)
{
    if (node==NULL)
    {
        return 0;
    }
    return size_Binarytree(node->left)+size_Binarytree(node->right)+2;
}


