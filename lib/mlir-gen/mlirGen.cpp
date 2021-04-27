#include "mlir-gen/mlirGen.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "PCC/PCCDialect.h"
#include "PCC/PCCOps.h"
#include "PCC/PCCTypes.h"

namespace {
class MLIRGenImpl {

public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
  mlir::ModuleOp mlirGen(ProtoCCParser::DocumentContext *ctx) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto net = mlir::pcc::NetworkType::get(
        builder.getContext(), mlir::pcc::NetworkType::Ordering::UNORDERED);
    
    auto set = mlir::pcc::SetType::get(net, 5);

    auto state = mlir::pcc::StateType::get(builder.getContext(), "I");

    mlir::pcc::FooOp fooOp = builder.create<mlir::pcc::FooOp>(builder.getUnknownLoc(), state);
    theModule.push_back(fooOp);
    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
};
} // namespace

namespace pcc {
mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx,
                       ProtoCCParser::DocumentContext *docCtx) {
  return MLIRGenImpl(mlirCtx).mlirGen(docCtx);
}
} // namespace pcc
