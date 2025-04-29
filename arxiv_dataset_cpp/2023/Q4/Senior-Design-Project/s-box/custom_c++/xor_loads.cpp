bool RISCVDAGToDAGISel::selectSBox(SDNode *Node) {
    SDValue LOAD0 = Node->getOperand(0);
    SDValue LOAD1 = Node->getOperand(1);
    //Check if there is a load pair
    if(LOAD0.getOpcode() != ISD::LOAD 
            || LOAD1.getOpcode() != ISD::LOAD)
      return false;
    SDValue LOAD0_op0 = LOAD0.getOperand(0);
    SDValue LOAD0_op1_offset = LOAD0.getOperand(1); // t0

    SDValue LOAD1_op0 = LOAD1.getOperand(0); 
    SDValue LOAD1_op1_offset = LOAD1.getOperand(1); 

    if(LOAD1_op1_offset.getOpcode() != ISD::ADD)
        return false;

    if(LOAD1_op0 != LOAD0_op0)
        return false;

   //Check if the addendum0 is the same as the second 
    SDValue LOAD1_op1_off_Addend0 = LOAD1_op1_offset.getOperand(0);
    if(LOAD1_op1_off_Addend0 != LOAD0_op1_offset)
        return false;

    SDValue LOAD1_op1_off_Addend1 = LOAD1_op1_offset.getOperand(1);

    auto *LOAD1_op1_offset_Addendum1C = 
        dyn_cast<ConstantSDNode>(LOAD1_op1_offset_Addendum1);
    if(!LOAD1_op1_offset_Addendum1C)
        return false;
    //Check if addendum1 is 16 more than first load offset
    if(LOAD1_op1_offset_Addendum1C->getZExtValue() != 16)
        return false;

    //Pattern is matched replace here

    return true;
}


