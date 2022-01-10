#include "FSM/FSMDialect.h"
#include "FSM/FSMOps.h"
#include "Support/LLVM.h"

using namespace mlir;
using namespace fsm;

#include "FSM/FSMDialect.cpp.inc"

void FSMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "FSM/FSMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "FSM/FSM.cpp.inc"
      >();
}