#include "spatialintegrationmethod.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SpatialIntegrationMethod::SpatialIntegrationMethod
   ( size_t       nx
   , size_t       ny
   , size_t       bufferWidth
   , double       dx
   , double       dy
   , t_boundary   boundary
   , double       gamma
)
   : nxProblem{ nx } , nyProblem{ ny }
   , bufferWidth{ bufferWidth }
   , dx{ dx }, dy{ dy }
   , boundary{ boundary }
   , gamma{ gamma }
   , nxFirst{ 0 +bufferWidth   }, nyFirst{ 0 +bufferWidth   }
   , nxLast { nx+bufferWidth   }, nyLast { ny+bufferWidth   }
   , nxTotal{ nx+bufferWidth*2 }, nyTotal{ ny+bufferWidth*2 }
{
   requireBoundaryInitialization = boundary.left   == BoundaryCondition::Dirichlet
                                || boundary.right  == BoundaryCondition::Dirichlet
                                || boundary.top    == BoundaryCondition::Dirichlet
                                || boundary.bottom == BoundaryCondition::Dirichlet;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SpatialIntegrationMethod::~SpatialIntegrationMethod
   (
){
   if( dirichletBoundaryLeft     ) freeVectors( dirichletBoundaryLeft    );
   if( dirichletBoundaryRight    ) freeVectors( dirichletBoundaryRight   );
   if( dirichletBoundaryTop      ) freeVectors( dirichletBoundaryTop     );
   if( dirichletBoundaryBottom   ) freeVectors( dirichletBoundaryBottom  );
}


void SpatialIntegrationMethod::initializeDirichletBoundaries
   ( t_matrices U
){
   dirichletBoundaryLeft   = createVectors( PRB_DIM, nyTotal );
   dirichletBoundaryRight  = createVectors( PRB_DIM, nyTotal );
   dirichletBoundaryTop    = createVectors( PRB_DIM, nxTotal );
   dirichletBoundaryBottom = createVectors( PRB_DIM, nxTotal );

   for( size_t k = 0; k < PRB_DIM; k++ ){
      for( size_t i = nxFirst; i < nxLast; i++ ){
         dirichletBoundaryBottom[k][i] = U[k][i][nyFirst];
         dirichletBoundaryTop[k][i]    = U[k][i][nyLast-1];
      }
      for( size_t j = nyFirst; j < nyLast; j++ ){
         dirichletBoundaryLeft[k][j]   = U[k][nxFirst][j];
         dirichletBoundaryRight[k][j]  = U[k][nxLast-1][j];
      }
   }

   requireBoundaryInitialization = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
size_t SpatialIntegrationMethod::requiredBufferWidth
   (
){
   return minimumBufferWidth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool SpatialIntegrationMethod::getCharacteristicsX
   ( t_matrices /*cx*/
   , t_matrices /*LUx*/
){
   return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool SpatialIntegrationMethod::getCharacteristicsY
   ( t_matrices /*cy*/
   , t_matrices /*LUy*/
){
   return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
t_status SpatialIntegrationMethod::applyBoundaryConditions
   ( t_matrices U
){
   if( requireBoundaryInitialization ){
      return { true, ReturnStatus::ErrorWrongParameter, "SpatialIntegrationMethod::applyBoundaryConditions: boundary not initialized." };
   }
   // left boundary
   switch( boundary.left ){
   case BoundaryCondition::Undefined:
      return { true, ReturnStatus::ErrorWrongParameter, "Unknown left boundary condition." };
      break;
   case BoundaryCondition::Periodic:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = bufferWidth; i > 0; i-- ){
            for( auto j = nyFirst; j < nyLast; j++ ){
               U[k][nxFirst-i][j] = U[k][nxLast-i][j];
            }
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = bufferWidth; i > 0; i-- ){
            for( auto j = nyFirst; j < nyLast; j++ ){
               U[k][nxFirst-i][j] = dirichletBoundaryLeft[k][j];
            }
         }
      }
      break;
   case BoundaryCondition::Neumann:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = bufferWidth; i > 0; i-- ){
            for( auto j = nyFirst; j < nyLast; j++ ){
               U[k][nxFirst-i][j] = U[k][nxFirst+i][j];
            }
         }
      }
      // use div B = 0 for dBx/dx and dBy/dy boundary conditions
      {
         double ratio_xy = dx/dy;
         for( auto i = nxFirst; i > 0; i-- ){
            for( auto j = nyFirst+1; j < nyLast-1; j++ ){
               U[4][i-1][j] = U[4][i+1][j] - ratio_xy*( U[5][i][j-1] - U[5][i][j+1] );
            }
         }
      }
      break;
   case BoundaryCondition::Open:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = bufferWidth; i > 0; i-- ){
            for( auto j = nyFirst; j < nyLast; j++ ){
               U[k][nxFirst-i][j] = U[k][nxFirst+i][j];
            }
         }
      }
      break;
   }

   // right boundary
   switch( boundary.right ){
   case BoundaryCondition::Undefined:
      return { true, ReturnStatus::ErrorWrongParameter, "Unknown right boundary condition." };
      break;
   case BoundaryCondition::Periodic:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = bufferWidth; i > 0; i-- ){
            for( auto j = nyFirst; j < nyLast; j++ ){
               U[k][nxLast-1+i][j] = U[k][nxFirst-1+i][j];
            }
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = bufferWidth; i > 0; i-- ){
            for( auto j = nyFirst; j < nyLast; j++ ){
               U[k][nxLast-1+i][j] = dirichletBoundaryRight[k][j];
            }
         }
      }
      break;
   case BoundaryCondition::Neumann:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = bufferWidth; i > 0; i-- ){
            for( auto j = nyFirst; j < nyLast; j++ ){
               U[k][nxLast-1+i][j] = U[k][nxLast-1-i][j];
            }
         }
      }
      // use div B = 0 for dBx/dx and dBy/dy boundary conditions
      {
         double ratio_xy = dx/dy;
         for( auto i = nxLast-1; i < nxTotal-1; i++ ){
            for( auto j = nyFirst+1; j < nyLast-1; j++ ){
               U[4][i+1][j] = U[4][i-1][j] - ratio_xy*( U[5][i][j+1] - U[5][i][j-1] );
            }
         }
      }
      break;
   case BoundaryCondition::Open:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = bufferWidth; i > 0; i-- ){
            for( auto j = nyFirst; j < nyLast; j++ ){
               U[k][nxLast-1+i][j] = U[k][nxLast-1-i][j];
            }
         }
      }
      break;
   }

   // bottom boundary
   switch( boundary.bottom ){
   case BoundaryCondition::Undefined:
      return { true, ReturnStatus::ErrorWrongParameter, "Unknown bottom boundary condition." };
      break;
   case BoundaryCondition::Periodic:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            for( auto j = bufferWidth; j > 0; j-- ){
               U[k][i][nyFirst-j] = U[k][i][nyLast-j];
            }
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            for( auto j = bufferWidth; j > 0; j-- ){
               U[k][i][nyFirst-j] = dirichletBoundaryBottom[k][i];
            }
         }
      }
      break;
   case BoundaryCondition::Neumann:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            for( auto j = bufferWidth; j > 0; j-- ){
               U[k][i][nyFirst-j] = U[k][i][nyFirst+j];
            }
         }
      }
      // use div B = 0 for dBx/dx and dBy/dy boundary conditions
      {
         double ratio_yx = dy/dx;
         for( auto i = nxFirst+1; i < nxLast-1; i++ ){
            for( auto j = nyFirst; j > 0; j-- ){
               U[5][i][j-1] = U[5][i][j+1] - ratio_yx*( U[4][i-1][j] - U[4][i+1][j] );
            }
         }
      }
      break;
   case BoundaryCondition::Open:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            for( auto j = bufferWidth; j > 0; j-- ){
               U[k][i][nyFirst-j] = U[k][i][nyFirst+j];
            }
         }
      }
      break;
   }

   // top boundary
   switch( boundary.top ){
   case BoundaryCondition::Undefined:
      return { true, ReturnStatus::ErrorWrongParameter, "Unknown top boundary condition." };
      break;
   case BoundaryCondition::Periodic:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            for( auto j = bufferWidth; j > 0; j-- ){
               U[k][i][nyLast-1+j] = U[k][i][nyFirst-1+j];
            }
         }
      }
      break;
   case BoundaryCondition::Dirichlet:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            for( auto j = bufferWidth; j > 0; j-- ){
               U[k][i][nyLast-1+j] = dirichletBoundaryTop[k][i];
            }
         }
      }
      break;
   case BoundaryCondition::Neumann:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            for( auto j = bufferWidth; j > 0; j-- ){
               U[k][i][nyLast-1+j] = U[k][i][nyLast-1-j];
            }
         }
      }
      // use div B = 0 for dBx/dx and dBy/dy boundary conditions
      {
         double ratio_yx = dy/dx;
         for( auto i = nxFirst+1; i < nxLast-1; i++ ){
            for( auto j = nyLast-1; j < nyTotal-1; j++ ){
               U[5][i][j+1] = U[5][i][j-1] - ratio_yx*( U[4][i+1][j] - U[4][i-1][j] );
            }
         }
      }
      break;
   case BoundaryCondition::Open:
      for( auto k = size_t{0}; k < PRB_DIM; k++ ){
         for( auto i = nxFirst; i < nxLast; i++ ){
            for( auto j = bufferWidth; j > 0; j-- ){
               U[k][i][nyLast-1+j] = U[k][i][nyLast-1-j];
            }
         }
      }
      break;
   }

   return { false, ReturnStatus::OK, "" };
}
