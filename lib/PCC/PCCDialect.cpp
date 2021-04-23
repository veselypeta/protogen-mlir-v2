#include "PCC/PCCOps.h"
#include <models/Expr.h>

using namespace mlir;
using namespace mlir::pcc;


void PCCDialect::initialize(){
    addOperations<
    #define GET_OP_LIST
    #include "PCC/PCC.cpp.inc"
    >();
    addTypes<
    #define GET_TYPEDEF_CLASSES
    #include "PCC/PCCTypes.cpp.inc"
    >();
}