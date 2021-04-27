
#include "PCC/PCCOps.h"
#include "mlir-gen/mlirGen.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <iostream>

namespace {
class MLIRGenImpl {

public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
  mlir::ModuleOp mlirGen(ProtoCCParser::DocumentContext *ctx, std::string compFile) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    // save the filename - used for location tracking
    filename = compFile;

    auto net = mlir::pcc::NetworkType::get(
        builder.getContext(), mlir::pcc::NetworkType::Ordering::UNORDERED);
    
    auto set = mlir::pcc::SetType::get(net, 5);

    auto state = mlir::pcc::StateType::get(builder.getContext(), "I");

    mlir::pcc::FooOp fooOp = builder.create<mlir::pcc::FooOp>(loc(*ctx->const_decl()[0]->ID()->getSymbol()), net);
    theModule.push_back(fooOp);
    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  std::string filename;

  mlir::Location loc(const antlr4::Token &tok) const {
    return mlir::FileLineColLoc::get(builder.getContext(), "file", tok.getLine(), tok.getCharPositionInLine());
  }
};
} // namespace

namespace pcc {
mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx,
                       ProtoCCParser::DocumentContext *docCtx, std::string filename) {
  return MLIRGenImpl(mlirCtx).mlirGen(docCtx, filename);
}
} // namespace pcc
