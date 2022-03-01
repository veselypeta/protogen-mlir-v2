#include <utility>

#include "FSM/FSMOps.h"
#include "mlir-gen/mlirGen.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <llvm/ADT/Optional.h>

using namespace mlir;
using namespace mlir::fsm;

namespace {
constexpr auto cache = "cache";
constexpr auto directory = "directory";
} // namespace

namespace {

class MLIRGenImpl {
private:
  ModuleOp theModule;
  OpBuilder builder;
  std::string filename;
  std::set<std::string> identStorage;
  std::string curMach;

  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

public:
  MLIRGenImpl(MLIRContext *ctx, std::string fileName)
      : builder(ctx), filename(std::move(fileName)) {}

  ModuleOp generate(ProtoCCParser::DocumentContext *ctx) {
    theModule = ModuleOp::create(builder.getUnknownLoc());

    builder.setInsertionPointToStart(theModule.getBody());

    SymbolTableScopeT global_scope(symbolTable);

    if (failed(mlirGen(ctx)))
      assert(0 && "failed to generate MLIR correctly!");

    if (failed(verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }
    return theModule;
  }

private:
  mlir::Location loc(const antlr4::Token &tok) const {
    return mlir::FileLineColLoc::get(
        mlir::Identifier::get(filename, builder.getContext()), tok.getLine(),
        tok.getCharPositionInLine());
  }

  mlir::Location loc(antlr4::tree::TerminalNode &terminal) const {
    return loc(*terminal.getSymbol());
  }

  LogicalResult declare(const std::string &ident, Value val) {
    // save the string to the class so that it won't be deleted, StrRef does not
    // own the data
    identStorage.insert(ident);
    // get a reference to the stored data
    // find will always succeed because we have inserted into the storage class
    llvm::StringRef identRef = *identStorage.find(ident);

    // shadowing not allowed!
    if (symbolTable.count(identRef) > 0) {
      assert(0 && "mlir value already declared in current scope!");
    }
    symbolTable.insert(identRef, val);
    return mlir::success();
  }

  mlir::Value lookup(const std::string &ident) {
    // Identifier must already be in the symbol table - otherwise error!
    if (!symbolTable.count(ident)) {
      return nullptr;
    }
    return symbolTable.lookup(ident);
  }

  mlir::Value lookupWithSymbols(const std::string &ident){
    if(auto symbolTableLookup = lookup(ident))
      return symbolTableLookup;

    // could be referring to a variable
    auto machOp = theModule.lookupSymbol<MachineOp>(curMach);
    for(auto varOp : machOp.getOps<VariableOp>())
      if(varOp.name() == ident)
        return varOp;


    // could be referring to a network
    for(auto netOp : theModule.getOps<NetworkOp>())
      if(netOp.sym_name().getLeafReference() == ident)
        return netOp;

    return nullptr;
  }

  size_t resolveRange(ProtoCCParser::Val_rangeContext *ctx) {
    if (auto intCtx = ctx->INT())
      return std::strtol(intCtx->getText().c_str(), nullptr, 10);
    else {
      Value constRef = lookup(ctx->ID()->getText());
      ConstOp defOp = constRef.getDefiningOp<ConstOp>();
      return defOp.value().cast<IntegerAttr>().getInt();
    }
  }

  LogicalResult mlirGen(ProtoCCParser::DocumentContext *ctx) {
    for (auto constCtx : ctx->const_decl())
      if (failed(mlirGen(constCtx)))
        return failure();

    for (auto initHw : ctx->init_hw())
      if (failed(mlirGen(initHw)))
        return failure();

    for (auto archBlock : ctx->arch_block())
      if (failed(mlirGen(archBlock)))
        return failure();

    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Const_declContext *ctx) {
    std::string value_str = ctx->INT()->getText();
    long size = std::strtol(value_str.c_str(), nullptr, 10);
    auto cOp = builder.create<ConstOp>(loc(*ctx->ID()), builder.getI64Type(),
                                       builder.getI64IntegerAttr(size));
    return declare(ctx->ID()->getText(), cOp);
  }

  LogicalResult mlirGen(ProtoCCParser::Init_hwContext *ctx) {
    if (failed(mlirGen(ctx->network_block())))
      return failure();
    if (failed(mlirGen(ctx->message_block())))
      return failure();
    if (failed(mlirGen(ctx->machines())))
      return success();
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Network_blockContext *ctx) {
    if (ctx == nullptr)
      return success();

    for (auto network : ctx->network_element()) {
      Location netLoc = loc(*network->ID());
      std::string netId = network->ID()->getText();
      std::string netOrd = network->element_type()->getText();
      std::transform(netOrd.begin(), netOrd.end(), netOrd.begin(), ::tolower);
      builder.create<NetworkOp>(netLoc, NetworkType::get(builder.getContext()),
                                netOrd, builder.getSymbolRefAttr(netId));
    }
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Message_blockContext *ctx) {
    if (ctx == nullptr)
      return success();
    std::string msgTypeId = ctx->ID()->getText();
    Location msgDeclLoc = loc(*ctx->ID());
    auto msgDecl = builder.create<MessageDecl>(msgDeclLoc, msgTypeId);
    Block *entry = msgDecl.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    for (auto decl : ctx->declarations())
      if (failed(mlirGenMsgDecl(decl)))
        return failure();
    if (ctx->declarations().empty())
      builder.create<NOPOp>(msgDeclLoc);
    builder.setInsertionPointAfter(msgDecl);
    return success();
  }

  LogicalResult mlirGenMsgDecl(ProtoCCParser::DeclarationsContext *ctx) {
    if (ctx == nullptr)
      return success();
    if (auto dataCtx = ctx->data_decl()) {
      Location l = loc(*dataCtx->ID());
      std::string id = dataCtx->ID()->getText();
      builder.create<MessageVariable>(l, DataType::get(builder.getContext()),
                                      id);
    }
    if (auto intDecl = ctx->int_decl()) {
      Location l = loc(*intDecl->ID());
      std::string id = intDecl->ID()->getText();
      size_t start_r = resolveRange(intDecl->range()->val_range(0));
      size_t end_r = resolveRange(intDecl->range()->val_range(1));
      builder.create<MessageVariable>(
          l, RangeType::get(builder.getContext(), start_r, end_r), id);
    }
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::MachinesContext *ctx) {
    if (ctx == nullptr)
      return success();
    if (auto cacheCtx = ctx->cache_block())
      if (failed(mlirGenMachDecl(cacheCtx->declarations(), cache,
                                 loc(*cacheCtx->ID()))))
        return failure();
    if (auto dirCtx = ctx->dir_block())
      if (failed(mlirGenMachDecl(dirCtx->declarations(), directory,
                                 loc(*dirCtx->ID()))))
        return failure();

    return success();
  }

  LogicalResult
  mlirGenMachDecl(const std::vector<ProtoCCParser::DeclarationsContext *> &ctx,
                  const std::string &machId, Location location) {
    auto machOp = builder.create<MachineOp>(location, machId,
                                            builder.getFunctionType({}, {}));
    builder.setInsertionPointToStart(machOp.addEntryBlock());

    for (auto decl : ctx) {
      if (auto stateDecl = decl->state_decl()) {
        Location stateLoc = loc(*stateDecl->STATE());
        std::string declId = stateDecl->STATE()->getText();
        std::string initState = stateDecl->ID()->getText();
        builder.create<VariableOp>(stateLoc,
                                   StateType::get(builder.getContext()),
                                   builder.getStringAttr(initState), declId);
      }
      if (auto dataDecl = decl->data_decl()) {
        Location dataDeclLoc = loc(*dataDecl->ID());
        std::string id = dataDecl->ID()->getText();
        builder.create<VariableOp>(
            dataDeclLoc, DataType::get(builder.getContext()), nullptr, id);
      }
      if (auto intDecl = decl->int_decl()) {
        Location intDeclLoc = loc(*intDecl->INTID());
        std::string id = intDecl->ID()->getText();
        size_t start_r = resolveRange(intDecl->range()->val_range(0));
        size_t end_r = resolveRange(intDecl->range()->val_range(1));
        auto initAttr =
            intDecl->INT()
                ? builder.getI64IntegerAttr(std::strtol(
                      intDecl->INT()->getText().c_str(), nullptr, 10))
                : nullptr;
        builder.create<VariableOp>(
            intDeclLoc, RangeType::get(builder.getContext(), start_r, end_r),
            initAttr, id);
      }
      if (auto idDecl = decl->id_decl()) {
        Location idLoc = loc(*idDecl->NID());
        auto get_id_decl_type =
            [&](ProtoCCParser::Id_declContext *ctx) -> Type {
          if (ctx->set_decl().empty())
            return IDType::get(builder.getContext());
          else
            return SetType::get(IDType::get(builder.getContext()),
                                resolveRange(idDecl->set_decl(0)->val_range()));
        };
        Type idType = get_id_decl_type(idDecl);
        builder.create<VariableOp>(idLoc, idType, nullptr,
                                   idDecl->ID(0)->getText());
      }
    }

    builder.setInsertionPointAfter(machOp);
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Arch_blockContext *ctx) {
    curMach = ctx->ID()->getText() == "cache" ? cache : directory;
    for (auto procBlock : ctx->arch_body()->process_block())
      if (failed(mlirGen(procBlock)))
        return failure();
    return success();
  }

  llvm::Optional<std::string>
  processFinalState(ProtoCCParser::Process_finalstateContext *ctx) {
    if (ctx == nullptr)
      return llvm::Optional<std::string>{};
    auto maybeFinalState = ctx->process_finalident()->ID();
    if (maybeFinalState == nullptr)
      return llvm::Optional<std::string>{};
    return llvm::Optional<std::string>{maybeFinalState->getText()};
  }

  LogicalResult mlirGen(ProtoCCParser::Process_blockContext *ctx) {
    if (ctx == nullptr)
      return success();

    // lookup the current machine
    MachineOp curMachOp = theModule.lookupSymbol<MachineOp>(curMach);
    builder.setInsertionPointToEnd(&curMachOp.getBody().front());

    // get data needed to construct the op
    auto procTransCtx = ctx->process_trans();
    std::string startState = procTransCtx->ID()->getText();
    std::string action = procTransCtx->process_events()->getText();
    auto finalState = processFinalState(procTransCtx->process_finalstate());
    auto nextStateAttr = finalState.hasValue()
                             ? builder.getSymbolRefAttr(finalState.getValue())
                             : nullptr;
    auto targetState = curMachOp.lookupSymbol<StateOp>(startState);
    if (targetState == nullptr) {
      targetState =
          builder.create<StateOp>(builder.getUnknownLoc(), startState);
      targetState.addEntryBlock();
    }
    builder.setInsertionPointToEnd(targetState.getBody());

    // create the type of the transition
    FunctionType transType;
    if (procTransCtx->process_events()->ID())
      transType =
          builder.getFunctionType({MsgType::get(builder.getContext())}, {});
    else
      transType = builder.getFunctionType({}, {});

    // create the transition
    Location procLoc = loc(*procTransCtx->ID());
    auto theTrans =
        builder.create<TransitionOp>(procLoc, action, transType, nextStateAttr);
    theTrans.addEntryBlock();

    // create the nested ops
    for (auto expr : ctx->process_expr())
      if (failed(mlirGen(expr)))
        return failure();

    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Process_exprContext *ctx) {
    if (auto exprCtx = ctx->expressions())
      if (failed(mlirGen(exprCtx)))
          return failure();
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::ExpressionsContext *ctx){
    if(auto assignCtx = ctx->assignment())
      return mlirGen(assignCtx);
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::AssignmentContext *ctx){
    // assignment is only used in two cases
    // 1 - for message construction i.e. msg = Request(...);
    // 2 = for updating the cache state cl=GetM_Ack_D.cl;
    auto lhsIdent = ctx->process_finalident()->getText();
    // case 2
    if(auto lhsValue = lookupWithSymbols(lhsIdent)){

    } else {
      assert(ctx->assign_types()->message_constr() != nullptr && "assignment must be to a message constructor");
      auto msgCtr = ctx->assign_types()->message_constr();
      Location msgLoc = loc(*msgCtr->ID());
      std::string msgType = msgCtr->ID()->getText();
      std::string msgName = msgCtr->message_expr(0)->object_expr()->getText();
      builder.create<MessageOp>(msgLoc,
                                MsgType::get(builder.getContext()),
                                builder.getSymbolRefAttr(msgType),
                                msgName,
                                llvm::None
                                );
    }
    return success();
  }
};

} // namespace

namespace mlir {
namespace fsm {
ModuleOp mlirGen(MLIRContext &mlirCtx, ProtoCCParser::DocumentContext *docCtx,
                 std::string filename) {
  return MLIRGenImpl(&mlirCtx, std::move(filename)).generate(docCtx);
}
} // namespace fsm
} // namespace mlir