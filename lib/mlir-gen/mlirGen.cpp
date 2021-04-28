#include "mlir-gen/mlirGen.h"
#include "PCC/PCCOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"

#include <iostream>
#include <set>
#include <string>

namespace {
class MLIRGenImpl {

public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
  mlir::ModuleOp mlirGen(ProtoCCParser::DocumentContext *ctx,
                         std::string compFile) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    // save the filename - used for location tracking
    filename = compFile;

    // set the insertion point to the start of the module
    builder.setInsertionPointToStart(theModule.getBody());

    // decalre a global scope
    SymbolTableScopeT global_scope(symbolTable);

    // recursivelly call mlirGen
    if (mlir::failed(mlirGen(ctx))) {
      return nullptr;
    }

    // initially create global scope

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

  std::set<std::string> identStorage;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

  mlir::Location loc(const antlr4::Token &tok) const {
    return mlir::FileLineColLoc::get(builder.getContext(), "file",
                                     tok.getLine(),
                                     tok.getCharPositionInLine());
  }

  // Used to declare MLIR Values along with their identifiers
  mlir::LogicalResult declare(std::string ident, mlir::Value val) {
    // save the string to the class so that it won't be deleted, StrRef does not own the data
    identStorage.insert(ident);
    // get a reference to the stored data
    // find will always succeed because we have inserted into the storage class
    llvm::StringRef identRef = *identStorage.find(ident);

    // shadowing not allowed!
    if (symbolTable.count(identRef) > 0) {
      assert(0 && "mlir value already delared in current scope!");
      return mlir::failure();
    }
    symbolTable.insert(identRef, val);
    return mlir::success();
  }

  mlir::Value lookup(std::string ident){
    if(!symbolTable.count(ident)){
      assert(0 && "attempting to lookup ident which is not declared!");
      return nullptr;
    }
    return symbolTable.lookup(ident);
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

    mlir::ConstantOp constOp = builder.create<mlir::ConstantOp>(
        constDeclLocation, constType, builder.getI64IntegerAttr(constValue));

    return declare(constId, constOp);
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
