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

#include <algorithm>
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

  // used to hold underlying data for llvm::StringRef
  std::set<std::string> identStorage;

  // A symbol table is used to hold (ident -> mlir::Value) references
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

  // return an mlir::Location object for builder operations
  mlir::Location loc(const antlr4::Token &tok) const {
    return mlir::FileLineColLoc::get(builder.getContext(), filename,
                                     tok.getLine(),
                                     tok.getCharPositionInLine());
  }

  // Used to declare MLIR Values along with their identifiers
  mlir::LogicalResult declare(std::string ident, mlir::Value val) {
    // save the string to the class so that it won't be deleted, StrRef does not
    // own the data
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

  mlir::Value lookup(std::string ident) {
    // Identifier must already be in the symbol table - otherwise error!
    if (!symbolTable.count(ident)) {
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

    // recursively call mlir on init_hw
    for (auto initHw : ctx->init_hw()) {
      if (mlir::failed(mlirGen(initHw))) {
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

  mlir::LogicalResult mlirGen(ProtoCCParser::Init_hwContext *ctx) {
    // recursivelly call for each network block
    if (mlir::failed(mlirGen(ctx->network_block())))
      return mlir::failure();
    // recursivelly call for each message block
    if (mlir::failed(mlirGen(ctx->message_block())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Network_blockContext *ctx) {
    // skip if the recursive call is nullptr
    if (ctx == nullptr)
      return mlir::success();
    for (auto network : ctx->network_element()) {
      std::string networkId = network->ID()->getText();
      mlir::Location netLoc = loc(*network->ID()->getSymbol());
      std::string orderingStr = network->element_type()->getText();
      // lowercase the string
      std::transform(orderingStr.begin(), orderingStr.end(),
                     orderingStr.begin(), ::tolower);
      // construct the appropriate type - ordered/unordered
      mlir::pcc::NetworkType netType = mlir::pcc::NetworkType::get(
          builder.getContext(),
          mlir::pcc::NetworkType::convertToOrder(orderingStr));

      // TODO - construct and appropriate op
      mlir::pcc::FooOp foo = builder.create<mlir::pcc::FooOp>(netLoc, netType);
      // Declare the network op in the scope
      if (failed(declare(networkId, foo)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Message_blockContext *ctx){
    // map decl ids to their MLIR type
    static std::map<std::string, mlir::Type> msgDecls;
    // skip if nullptr
    if(ctx == nullptr)
      return mlir::success();
    std::string msgId = ctx->ID()->getText();
    mlir::Location msgLoc = loc(*ctx->ID()->getSymbol());
    for(auto msgDecCtx : ctx->declarations()){
      if (msgDecCtx->int_decl())
        mlirTypeGen(msgDecCtx->int_decl());
      msgDecCtx->bool_decl();
      msgDecCtx->state_decl();
      msgDecCtx->data_decl();
      msgDecCtx->id_decl();
    }
    return mlir::success();
  }

  mlir::pcc::PCCType mlirTypeGen(ProtoCCParser::Int_declContext *ctx){
    // Integer declarations always define an integer range
    // we find the start and stop of the integer range
    std::string intDeclId = ctx->ID()->getText();

    // get a reference to the start and end sub-range decls
    auto startRange = ctx->range()->val_range()[0];
    auto endRange = ctx->range()->val_range()[1];

    // determine the value of each sub-range
    size_t startRangeVal;
    if(startRange->INT()){
      startRangeVal = std::atoi(startRange->INT()->getText().c_str());
    } else {
      std::string constRef = startRange->ID()->getText();
      // lookup the reference in the scope
      mlir::Value value = lookup(constRef);
      if(!value.getType().isa<mlir::IntegerType>())
        assert(0 && "const ref lookup returned non integer type!");
       startRangeVal = static_cast<mlir::ConstantOp>(value.getDefiningOp()).valueAttr().cast<mlir::IntegerAttr>().getInt();
    }

    size_t endRangeVal;
    if(endRange->INT()){
      endRangeVal = std::atoi(endRange->INT()->getText().c_str());
    } else {
      std::string constRef = endRange->ID()->getText();
      // lookup the reference in the scope
      mlir::Value value = lookup(constRef);
      if(!value.getType().isa<mlir::IntegerType>())
        assert(0 && "const ref lookup returned non integer type!");
      endRangeVal = static_cast<mlir::ConstantOp>(value.getDefiningOp()).valueAttr().cast<mlir::IntegerAttr>().getInt();
    }

    // build the type - custom type with a sub-range
    auto rangeType =  mlir::pcc::IntRangeType::get(builder.getContext(), startRangeVal, endRangeVal);
    builder.create<mlir::pcc::FooOp>(loc(*startRange->INT()->getSymbol()), rangeType);
    return rangeType;
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
