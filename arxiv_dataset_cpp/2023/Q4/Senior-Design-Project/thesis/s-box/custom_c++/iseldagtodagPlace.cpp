void RISCVDAGToDAGISel::Select(SDNode *Node) {
    .
}
    .
switch (Opcode) {
    .
    .
    case ISD::XOR:{
        if (tryShrinkShlLogicImm(Node))
          return;
        if (selectSbox(Node))
          return;
        break;
    }
}
