
#include "mlir-gen/mlirGen.h"
#include "PCC/PCCOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include <iostream>

namespace {
class MLIRGenImpl {

public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
  mlir::ModuleOp mlirGen(ProtoCCParser::DocumentContext *ctx,
                         std::string compFile) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    // save the filename - used for location tracking
    filename = compFile;

    // auto net = mlir::pcc::NetworkType::get(
    //     builder.getContext(), mlir::pcc::NetworkType::Ordering::UNORDERED);

    // auto set = mlir::pcc::SetType::get(net, 5);

    // auto state = mlir::pcc::StateType::get(builder.getContext(), "I");

    // mlir::pcc::FooOp fooOp =
    // builder.create<mlir::pcc::FooOp>(loc(*ctx->const_decl()[0]->ID()->getSymbol()),
    // net); theModule.push_back(fooOp); set the insertion point to the start of
    // the module
    builder.setInsertionPointToStart(theModule.getBody());

    // recursivelly call mlirGen
    if (mlir::failed(mlirGen(ctx))){
      return nullptr;
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the PCC operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  std::string filename;

  mlir::Location loc(const antlr4::Token &tok) const {
    return mlir::FileLineColLoc::get(builder.getContext(), "file",
                                     tok.getLine(),
                                     tok.getCharPositionInLine());
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::DocumentContext *ctx) {
    // recursivelly call mlirGen on const_decl operations
    for (auto constCtx : ctx->const_decl()) {
      if (mlir::failed(mlirGen(constCtx))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Const_declContext *ctx) {
    std::string constId = ctx->ID()->getText();
    int constValue = std::atoi(ctx->INT()->getText().c_str());
    mlir::IntegerType constType = builder.getI64Type();
    mlir::Location constDeclLocation = loc(*ctx->ID()->getSymbol());

    // build a const decl op - TODO
    builder.create<mlir::ConstantOp>(constDeclLocation, constType, builder.getI64IntegerAttr(constValue));

    return mlir::success();
  }
};
} // namespace

namespace pcc {
mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx,
                       ProtoCCParser::DocumentContext *docCtx,
                       std::string filename) {
  return MLIRGenImpl(mlirCtx).mlirGen(docCtx, filename);
}
} // namespace pcc
