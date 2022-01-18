#include "FSM/FSMDialect.h"
#include "FSM/FSMOps.h"
#include "Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace fsm;

#include "FSM/FSMDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// FSMInlinerInterface
//===----------------------------------------------------------------------===//

struct FSMInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // All ops of FSM can legally be inlined
  bool isLegalToInline(Operation * /*call*/, Operation * /*callable*/,
                       bool /*wouldBeCloned*/) const override {
    return true;
  }
  bool isLegalToInline(Region * /*dest*/, Region * /*src*/,
                       bool /*wouldBeCloned*/,
                       BlockAndValueMapping & /*valueMapping*/) const override {
    return true;
  }

  bool isLegalToInline(Operation * /*op*/, Region * /*dest*/,
                       bool /*wouldBeCloned*/,
                       BlockAndValueMapping & /*valueMapping*/) const override {
    return true;
  }

  void handleTerminator(Operation * /*op*/,
                        ArrayRef<Value> /*valuesToReplace*/) const override {
    // Do nothing for now;
    // the only terminator is break; and it returns to values
  }
};

void FSMDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "FSM/FSM.cpp.inc"
      >();

  addInterfaces<FSMInlinerInterface>();
}