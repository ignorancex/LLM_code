#include "enums.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
WriteSkipMode fromString<WriteSkipMode>
   ( std::string name
){
   WriteSkipMode member;
   std::transform( name.begin(), name.end(), name.begin(), ::tolower );

   if( false ){ // just to align the ifs below
   } else if( name == "step" ){
      member = WriteSkipMode::Step;
   } else if( name == "time" ){
      member = WriteSkipMode::Time;
   } else {
      member = WriteSkipMode::Undefined;
   }

   return member;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
std::string toString<WriteSkipMode>
   ( WriteSkipMode member
){
   std::string name;

   switch( member ){
   case WriteSkipMode::Undefined:
      name = "undefined";
      break;
   case WriteSkipMode::Step:
      name = "step";
      break;
   case WriteSkipMode::Time:
      name = "time";
      break;
   }

   return name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
WriteFileCountMode fromString<WriteFileCountMode>
   ( std::string name
){
   WriteFileCountMode member;
   std::transform( name.begin(), name.end(), name.begin(), ::tolower );

   if( false ){ // just to align the ifs below
   } else if( name == "single" ){
      member = WriteFileCountMode::Single;
   } else if( name == "multiple" ){
      member = WriteFileCountMode::Multiple;
   } else {
      member = WriteFileCountMode::Undefined;
   }

   return member;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
std::string toString<WriteFileCountMode>
   ( WriteFileCountMode member
){
   std::string name;

   switch( member ){
   case WriteFileCountMode::Undefined:
      name = "undefined";
      break;
   case WriteFileCountMode::Single:
      name = "single";
      break;
   case WriteFileCountMode::Multiple:
      name = "multiple";
      break;
   }

   return name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
TimeStepMode fromString<TimeStepMode>
   ( std::string name
){
   TimeStepMode member;
   std::transform( name.begin(), name.end(), name.begin(), ::tolower );

   if( false ){ // just to align the ifs below
   } else if( name == "constant" ){
      member = TimeStepMode::Constant;
   } else if( name == "variable" ){
      member = TimeStepMode::Variable;
   } else {
      member = TimeStepMode::Undefined;
   }

   return member;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
std::string toString<TimeStepMode>
   ( TimeStepMode member
){
   std::string name;

   switch( member ){
   case TimeStepMode::Undefined:
      name = "undefined";
      break;
   case TimeStepMode::Constant:
      name = "constant";
      break;
   case TimeStepMode::Variable:
      name = "variable";
      break;
   }

   return name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
TimeStepMethod fromString<TimeStepMethod>
   ( std::string name
){
   TimeStepMethod member;
   std::transform( name.begin(), name.end(), name.begin(), ::tolower );

   if( false ){ // just to align the ifs below
   } else if( name == "euler" ){
      member = TimeStepMethod::Euler;
   } else if( name == "rk3" ){
      member = TimeStepMethod::RungeKutta3_TVD;
   } else {
      member = TimeStepMethod::Undefined;
   }

   return member;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
std::string toString<TimeStepMethod>
   ( TimeStepMethod member
){
   std::string name;

   switch( member ){
   case TimeStepMethod::Undefined:
      name = "undefined";
      break;
   case TimeStepMethod::Euler:
      name = "euler";
      break;
   case TimeStepMethod::RungeKutta3_TVD:
      name = "rk3";
      break;
   }

   return name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
IntegrationMethod fromString<IntegrationMethod>
   ( std::string name
){
   IntegrationMethod member;
   std::transform( name.begin(), name.end(), name.begin(), ::tolower );

   if( false ){ // just to align the ifs below
   } else if( name == "central fd" ){
      member = IntegrationMethod::CentralFD;
   } else if( name == "eno-roe" ){
      member = IntegrationMethod::ENO_Roe;
   } else if( name == "eno-lf" ){
      member = IntegrationMethod::ENO_LF;
   } else {
      member = IntegrationMethod::Undefined;
   }

   return member;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
std::string toString<IntegrationMethod>
   ( IntegrationMethod member
){
   std::string name;

   switch( member ){
   case IntegrationMethod::Undefined:
      name = "undefined";
      break;
   case IntegrationMethod::CentralFD:
      name = "central fd";
      break;
   case IntegrationMethod::ENO_Roe:
      name = "eno-roe";
      break;
   case IntegrationMethod::ENO_LF:
      name = "eno-lf";
      break;
   }

   return name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
BoundaryCondition fromString<BoundaryCondition>
   ( std::string name
){
   BoundaryCondition member;
   std::transform( name.begin(), name.end(), name.begin(), ::tolower );

   if( false ){ // just to align the ifs below
   } else if( name == "periodic" ){
      member = BoundaryCondition::Periodic;
   } else if( name == "dirichlet" ){
      member = BoundaryCondition::Dirichlet;
   } else if( name == "neumann" ){
      member = BoundaryCondition::Neumann;
   } else if( name == "open" ){
      member = BoundaryCondition::Open;
   } else {
      member = BoundaryCondition::Undefined;
   }

   return member;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
std::string toString<BoundaryCondition>
   ( BoundaryCondition member
){
   std::string name;

   switch( member ){
   case BoundaryCondition::Undefined:
      name = "undefined";
      break;
   case BoundaryCondition::Periodic:
      name = "periodic";
      break;
   case BoundaryCondition::Dirichlet:
      name = "dirichlet";
      break;
   case BoundaryCondition::Neumann:
      name = "neumann";
      break;
   case BoundaryCondition::Open:
      name = "open";
      break;
   }

   return name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
DivBCorrectionMethod fromString<DivBCorrectionMethod>
   ( std::string name
){
   DivBCorrectionMethod member;
   std::transform( name.begin(), name.end(), name.begin(), ::tolower );

   if( false ){ // just to align the ifs below
   } else if( name == "sor" ){
      member = DivBCorrectionMethod::SOR;
   } else {
      member = DivBCorrectionMethod::Undefined;
   }

   return member;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
std::string toString<DivBCorrectionMethod>
   ( DivBCorrectionMethod member
){
   std::string name;

   switch( member ){
   case DivBCorrectionMethod::Undefined:
      name = "undefined";
      break;
   case DivBCorrectionMethod::SOR:
      name = "SOR";
      break;
   }

   return name;
}
