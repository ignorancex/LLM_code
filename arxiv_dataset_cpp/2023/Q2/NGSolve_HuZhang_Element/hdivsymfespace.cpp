#include <comp.hpp>  
#include "hdivsym.hpp"
#include "hdivsymfespace.hpp"
#include "myDiffOp.hpp"


namespace ngcomp
{

  HDivSymFESpace :: HDivSymFESpace (shared_ptr<MeshAccess> ama, const Flags & flags)
    : FESpace (ama, flags)
  {
    type = "HDivSymFESpace";


    // default = 3
    order = int(flags.GetNumFlag ("order", 3));
    evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpIdHDivSym>>();
    flux_evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpDivHDivSym>>();
  }
    
  void HDivSymFESpace :: Update()
  {
    int n_vert = ma->GetNV();  
    int n_edge = ma->GetNEdges(); 
    int n_cell = ma->GetNE();  

    first_edge_dof.SetSize (n_edge+1);
    int ii = 3*n_vert;
    for (int i = 0; i < n_edge; i++, ii+=2*(order-1))
      first_edge_dof[i] = ii;
    first_edge_dof[n_edge] = ii;
      
    first_cell_dof.SetSize (n_cell+1);
    for (int i = 0; i < n_cell; i++, ii+=3*(order-1)*(order-2)/2 + 3*(order-1))
      first_cell_dof[i] = ii;
    first_cell_dof[n_cell] = ii;

//     cout << "first_edge_dof = " << endl << first_edge_dof << endl;
//     cout << "first_cell_dof = " << endl << first_cell_dof << endl;

    SetNDof (ii);
  }

  void HDivSymFESpace :: GetDofNrs (ElementId ei, Array<DofId> & dnums) const
  {
    // returns dofs of element number elnr
    dnums.SetSize(0);
    auto ngel = ma->GetElement (ei);

    // vertex dofs
    for (auto v : ngel.Vertices()) {
      dnums.Append(3*v);
      dnums.Append(3*v+1);
      dnums.Append(3*v+2);
    }

    // edge dofs
    for (auto e : ngel.Edges())
      for(auto j : Range(first_edge_dof[e], first_edge_dof[e+1]))
        dnums.Append (j);

    // inner dofs
    if (ei.IsVolume())
      for(auto j : Range(first_cell_dof[ei.Nr()], first_cell_dof[ei.Nr()+1]))
        dnums.Append (j);
  }

  
  FiniteElement & HDivSymFESpace :: GetFE (ElementId ei, Allocator & alloc) const
  {
    auto ngel = ma->GetElement (ei);
    switch (ngel.GetType())
      {
      case ET_TRIG:
        {
          auto trig = new (alloc) HDivSymFE(order);
          trig->SetVertexNumbers (ngel.vertices);
          return *trig;
        }
      default:
        throw Exception (string("Element type ")+ToString(ngel.GetType())+" not supported");
      }
  }
    
    
        

  
}

void ExportHDivSymFESpace(py::module m)
{
  using namespace ngcomp;

  cout << "called ExportHDivSymFESpace" << endl;

  ExportFESpace<HDivSymFESpace>(m, "HDivSymFESpace", true)
    ;
}
