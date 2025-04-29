// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for flyinglightspeck/ReadFLSRequest
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
#include "flyinglightspeck/ReadFLS.h"
#include "visibility_control.h"
#include "ROSPubSubTemplates.hpp"
#include "ROSServiceTemplates.hpp"
class FLYINGLIGHTSPECK_EXPORT flyinglightspeck_msg_ReadFLSRequest_common : public MATLABROSMsgInterface<flyinglightspeck::ReadFLS::Request> {
  public:
    virtual ~flyinglightspeck_msg_ReadFLSRequest_common(){}
    virtual void copy_from_struct(flyinglightspeck::ReadFLS::Request* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const flyinglightspeck::ReadFLS::Request* msg, MultiLibLoader loader, size_t size = 1);
};
  void flyinglightspeck_msg_ReadFLSRequest_common::copy_from_struct(flyinglightspeck::ReadFLS::Request* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //str
        const matlab::data::CharArray str_arr = arr["Str"];
        msg->str = str_arr.toAscii();
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Str' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Str' is wrong type; expected a string.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T flyinglightspeck_msg_ReadFLSRequest_common::get_arr(MDFactory_T& factory, const flyinglightspeck::ReadFLS::Request* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Str"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("flyinglightspeck/ReadFLSRequest");
    // str
    auto currentElement_str = (msg + ctr)->str;
    outArray[ctr]["Str"] = factory.createCharArray(currentElement_str);
    }
    return std::move(outArray);
  }
class FLYINGLIGHTSPECK_EXPORT flyinglightspeck_msg_ReadFLSResponse_common : public MATLABROSMsgInterface<flyinglightspeck::ReadFLS::Response> {
  public:
    virtual ~flyinglightspeck_msg_ReadFLSResponse_common(){}
    virtual void copy_from_struct(flyinglightspeck::ReadFLS::Response* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const flyinglightspeck::ReadFLS::Response* msg, MultiLibLoader loader, size_t size = 1);
};
  void flyinglightspeck_msg_ReadFLSResponse_common::copy_from_struct(flyinglightspeck::ReadFLS::Response* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //str
        const matlab::data::CharArray str_arr = arr["Str"];
        msg->str = str_arr.toAscii();
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Str' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Str' is wrong type; expected a string.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T flyinglightspeck_msg_ReadFLSResponse_common::get_arr(MDFactory_T& factory, const flyinglightspeck::ReadFLS::Response* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Str"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("flyinglightspeck/ReadFLSResponse");
    // str
    auto currentElement_str = (msg + ctr)->str;
    outArray[ctr]["Str"] = factory.createCharArray(currentElement_str);
    }
    return std::move(outArray);
  } 
class FLYINGLIGHTSPECK_EXPORT flyinglightspeck_ReadFLS_service : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~flyinglightspeck_ReadFLS_service(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
    virtual std::shared_ptr<MATLABSvcServerInterface> generateSvcServerInterface();
    virtual std::shared_ptr<MATLABSvcClientInterface> generateSvcClientInterface();
};  
  std::shared_ptr<MATLABPublisherInterface> 
          flyinglightspeck_ReadFLS_service::generatePublisherInterface(ElementType type){
    std::shared_ptr<MATLABPublisherInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSPublisherImpl<flyinglightspeck::ReadFLS::Request,flyinglightspeck_msg_ReadFLSRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSPublisherImpl<flyinglightspeck::ReadFLS::Response,flyinglightspeck_msg_ReadFLSResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSubscriberInterface> 
          flyinglightspeck_ReadFLS_service::generateSubscriberInterface(ElementType type){
    std::shared_ptr<MATLABSubscriberInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSSubscriberImpl<flyinglightspeck::ReadFLS::Request,flyinglightspeck::ReadFLS::Request::ConstPtr,flyinglightspeck_msg_ReadFLSRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSSubscriberImpl<flyinglightspeck::ReadFLS::Response,flyinglightspeck::ReadFLS::Response::ConstPtr,flyinglightspeck_msg_ReadFLSResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSvcServerInterface> 
          flyinglightspeck_ReadFLS_service::generateSvcServerInterface(){
    return std::make_shared<ROSSvcServerImpl<flyinglightspeck::ReadFLS::Request,flyinglightspeck::ReadFLS::Response,flyinglightspeck_msg_ReadFLSRequest_common,flyinglightspeck_msg_ReadFLSResponse_common>>();
  }
  std::shared_ptr<MATLABSvcClientInterface> 
          flyinglightspeck_ReadFLS_service::generateSvcClientInterface(){
    return std::make_shared<ROSSvcClientImpl<flyinglightspeck::ReadFLS,flyinglightspeck::ReadFLS::Request,flyinglightspeck::ReadFLS::Response,flyinglightspeck_msg_ReadFLSRequest_common,flyinglightspeck_msg_ReadFLSResponse_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface> 
          flyinglightspeck_ReadFLS_service::generateRosbagWriterInterface(ElementType type){
    std::shared_ptr<MATLABRosbagWriterInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSBagWriterImpl<flyinglightspeck::ReadFLSRequest,flyinglightspeck_msg_ReadFLSRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSBagWriterImpl<flyinglightspeck::ReadFLSResponse,flyinglightspeck_msg_ReadFLSResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(flyinglightspeck_msg_ReadFLSRequest_common, MATLABROSMsgInterface<flyinglightspeck::ReadFLSRequest>)
CLASS_LOADER_REGISTER_CLASS(flyinglightspeck_msg_ReadFLSResponse_common, MATLABROSMsgInterface<flyinglightspeck::ReadFLSResponse>)
CLASS_LOADER_REGISTER_CLASS(flyinglightspeck_ReadFLS_service, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1
