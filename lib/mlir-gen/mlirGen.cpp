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

  mlir::LogicalResult mlirGen(ProtoCCParser::Message_blockContext *ctx) {
    // skip if nullptr
    if (ctx == nullptr)
      return mlir::success();

    // map decl ids to their MLIR type
    static std::map<std::string, mlir::Type> msgDecls;

    std::string msgId = ctx->ID()->getText();
    mlir::Location msgLoc = loc(*ctx->ID()->getSymbol());
    for (auto msgDecCtx : ctx->declarations()) {
      if (msgDecCtx->int_decl()) {
        auto idTypePair = mlirTypeGen(msgDecCtx->int_decl());
        msgDecls.insert(idTypePair);
      }

      if (msgDecCtx->bool_decl()) {
        auto idBoolTypePair = mlirTypeGen(msgDecCtx->bool_decl());
        msgDecls.insert(idBoolTypePair);
      }

      if (msgDecCtx->state_decl()) {
        auto idStateTypePair = mlirTypeGen(msgDecCtx->state_decl());
        msgDecls.insert(idStateTypePair);
      }

      if (msgDecCtx->data_decl()){
        auto idDataTypePair = mlirTypeGen(msgDecCtx->data_decl());
        msgDecls.insert(idDataTypePair);
      }

      if (msgDecCtx->id_decl()) {
        auto idTypePair = mlirTypeGen(msgDecCtx->id_decl());
        msgDecls.insert(idTypePair);
      }
    }
    return mlir::success();
  }

  std::pair<std::string, mlir::pcc::IntRangeType>
  mlirTypeGen(ProtoCCParser::Int_declContext *ctx) {
    // Integer declarations always define an integer range
    // we find the start and stop of the integer range
    std::string intDeclId = ctx->ID()->getText();

    // get a reference to the start and end sub-range decls
    auto startRange = ctx->range()->val_range()[0];
    auto endRange = ctx->range()->val_range()[1];

    // determine the value of each sub-range
    size_t startRangeVal = getIntFromValRange(startRange);
    size_t endRangeVal = getIntFromValRange(endRange);

    // build the type - custom type with a sub-range
    auto rangeType = mlir::pcc::IntRangeType::get(builder.getContext(),
                                                  startRangeVal, endRangeVal);
    return std::make_pair(intDeclId, rangeType);
  }

  std::pair<std::string, mlir::IntegerType>
  mlirTypeGen(ProtoCCParser::Bool_declContext *ctx) {
    std::string declId = ctx->ID()->getText();
    mlir::IntegerType boolType = builder.getI1Type();
    return std::make_pair(declId, boolType);
  }

  std::pair<std::string, mlir::pcc::StateType>
  mlirTypeGen(ProtoCCParser::State_declContext *ctx) {
    // a state declaration is always identified through the keyword State
    std::string declId = "State";
    std::string stateInitializer = ctx->ID()->getText();
    mlir::pcc::StateType stateType =
        mlir::pcc::StateType::get(builder.getContext(), stateInitializer);
    return std::make_pair(declId, stateType);
  }

  // Does not support ID initial assignment i.e. cl ID = //something;
  std::pair<std::string, mlir::pcc::PCCType>
  mlirTypeGen(ProtoCCParser::Id_declContext *ctx) {
    assert(!ctx->ID().empty() && "id decl had no ids defined");
    std::string declId = ctx->ID()[0]->getText();
    mlir::pcc::IDType idType = mlir::pcc::IDType::get(builder.getContext());
    // if set
    if (!ctx->set_decl().empty()) {
      auto firstSet = ctx->set_decl()[0];
      size_t setSize = getIntFromValRange(firstSet->val_range());
      mlir::pcc::SetType setType = mlir::pcc::SetType::get(idType, setSize);
      return std::make_pair(declId, setType);
    }
    return std::make_pair(declId, idType);
  }

  std::pair<std::string, mlir::pcc::DataType>
  mlirTypeGen(ProtoCCParser::Data_declContext *ctx) {
    std::string declId = ctx->ID()->getText();
    mlir::pcc::DataType dataType =
        mlir::pcc::DataType::get(builder.getContext());
    return std::make_pair(declId, dataType);
  }

  // get integer value from value range
  size_t getIntFromValRange(ProtoCCParser::Val_rangeContext *ctx) {
    if (ctx == nullptr)
      assert(0 && "val range context was nullptr");

    // if INT easy, convert to integer
    if (ctx->INT())
      return std::atoi(ctx->INT()->getText().c_str());

    // if ID we need to lookup in symbol table
    std::string constRef = ctx->ID()->getText();
    mlir::Value valRef = lookup(constRef);
    // we know this must have come from a ConstantOp
    mlir::ConstantOp constOp =
        static_cast<mlir::ConstantOp>(valRef.getDefiningOp());
    // extract the value attribute
    mlir::IntegerAttr intAttr = constOp.getValue().cast<mlir::IntegerAttr>();
    // return the int value
    return intAttr.getInt();
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
