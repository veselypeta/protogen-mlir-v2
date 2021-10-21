#include "PCC/PCCOps.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::pcc;

#include "PCC/PCCDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PCCInlinerInterface
//===----------------------------------------------------------------------===//

struct PCCInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // All ops of PCC can legally be inlined
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const override {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned, BlockAndValueMapping &valueMapping) const override{
    return true;
  }

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const override {
    // Do nothing for now;
    // the only terminator is break; and it returns to values
  }
};

void PCCDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "PCC/PCC.cpp.inc"
      >();
  addInterfaces<PCCInlinerInterface>();
}