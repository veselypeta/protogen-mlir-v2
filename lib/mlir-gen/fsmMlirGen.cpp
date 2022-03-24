#include <utility>

#include "FSM/FSMOps.h"
#include "mlir-gen/mlirGen.h"

#include "FSM/FSMUtils.h"
#include "ProtoCCBaseVisitor.h"
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
  std::set<std::string> stableStates;

  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

  std::map<std::string, std::string> msgNameTypeMap;

public:
  MLIRGenImpl(MLIRContext *ctx, std::string fileName)
      : builder(ctx), filename(std::move(fileName)) {}

  ModuleOp generate(ProtoCCParser::DocumentContext *ctx) {
    preprocessMsgNames(ctx);
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
  void preprocessMsgNames(ProtoCCParser::DocumentContext *ctx) {
    class MsgConstructorVisitor : public ProtoCCBaseVisitor {
    public:
      antlrcpp::Any
      visitMessage_constr(ProtoCCParser::Message_constrContext *ctx) override {
        std::string msgName = ctx->message_expr().at(0)->getText();
        std::string msgType = ctx->ID()->getText();
        msgNameMap.insert({msgName, msgType});
        return visitChildren(ctx);
      }
      std::map<std::string, std::string> msgNameMap;
    };

    MsgConstructorVisitor msgVisitor;
    ctx->accept(&msgVisitor);
    msgNameTypeMap = msgVisitor.msgNameMap;
  }

  mlir::Location loc(const antlr4::Token &tok) const {
    return mlir::FileLineColLoc::get(
        mlir::Identifier::get(filename, builder.getContext()), tok.getLine(),
        tok.getCharPositionInLine() + 1);
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

    //    // shadowing not allowed!
    //    if (symbolTable.count(identRef) > 0) {
    //      assert(0 && "mlir value already declared in current scope!");
    //    }
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

  mlir::Value lookupWithSymbols(const std::string &ident) {
    if (auto symbolTableLookup = lookup(ident))
      return symbolTableLookup;

    // could be referring to a variable
    auto machOp = theModule.lookupSymbol<MachineOp>(curMach);
    for (auto varOp : machOp.getOps<VariableOp>())
      if (varOp.name() == ident)
        return varOp;

    // could be referring to a network
    for (auto netOp : theModule.getOps<NetworkOp>())
      if (netOp.sym_name().getLeafReference() == ident)
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

  Value mlirGen(ProtoCCParser::Val_rangeContext *ctx) {
    if (auto intCtx = ctx->INT()) {
      return getIntegerConstant(intCtx);
    } else {
      return lookupWithSymbols(ctx->ID()->getText());
    }
  }

  Value getIntegerConstant(antlr4::tree::TerminalNode *node) {
    return builder.create<ConstOp>(loc(*node), builder.getI64Type(),
                                   builder.getI64IntegerAttr(std::strtol(
                                       node->getText().c_str(), nullptr, 10)));
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
    return declare(ctx->ID()->getText(), getIntegerConstant(ctx->INT()));
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
      auto networkOp = builder.create<NetworkOp>(
          netLoc, NetworkType::get(builder.getContext()), netOrd,
          builder.getSymbolRefAttr(netId));
      assert(succeeded(declare(netId, networkOp)));
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
    // add stable states
    for (auto ss : ctx->arch_body()->stable_def()->ID())
      stableStates.insert(ss->getText());

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
    SymbolTableScopeT proc_scope(symbolTable);
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
    builder.setInsertionPointToStart(theTrans.addEntryBlock());

    // add the msg param to the scope
    if (theTrans.getNumFuncArguments() == 1)
      assert(succeeded(declare(action, theTrans.getArgument(0))) &&
             "failed to add msg param to scope");

    // create the nested ops
    if (ctx->process_expr().empty()) {
      builder.create<NOPOp>(builder.getUnknownLoc());
    }
    for (auto expr : ctx->process_expr())
      if (failed(mlirGen(expr)))
        return failure();

    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Process_exprContext *ctx) {
    if (auto exprCtx = ctx->expressions())
      if (failed(mlirGen(exprCtx)))
        return failure();
    if (auto transCtx = ctx->transaction())
      if (failed(mlirGen(transCtx)))
        return failure();
    if (auto sendCtx = ctx->network_send())
      return mlirGen(sendCtx);

    if (auto mcastCtx = ctx->network_mcast())
      return mlirGen(mcastCtx);

    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Network_sendContext *ctx) {
    Location location = loc(*ctx->ID(0));
    auto netId = ctx->ID(0)->getText();
    auto msgIdent = ctx->ID(1)->getText();
    auto network = lookup(netId);
    auto msg = lookup(msgIdent);
    builder.create<SendOp>(location, network, msg);
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Network_mcastContext *ctx) {
    Location l = loc(*ctx->ID(0));
    auto netId = ctx->ID(0)->getText();
    auto msgId = ctx->ID(1)->getText();
    auto setId = ctx->ID(2)->getText();

    auto network = lookup(netId);
    auto msg = lookup(msgId);
    auto set = lookupWithSymbols(setId);
    builder.create<MulticastOp>(l, network, msg, set);
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::ExpressionsContext *ctx) {
    // assignment
    if (auto assignCtx = ctx->assignment())
      return mlirGen(assignCtx);

    // conditional
    if (auto condCtx = ctx->conditional())
      return mlirGen(condCtx);

    // object_block
    if (auto objBlockCtx = ctx->object_block()) {
      mlirGen(objBlockCtx->object_expr());
      return success();
    }

    // set_block
    if (auto setCtx = ctx->set_block()) {
      mlirGen(setCtx->set_func());
      return success();
    }

    // internal_event_block - unsupported
    if (ctx->internal_event_block() != nullptr)
      assert(0 && "internal event blocks are unsupported");
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::ConditionalContext *ctx) {
    if (auto ifCtx = ctx->if_stmt()) {
      auto conditional = mlirGen(ifCtx->cond_comb());
      auto elseRegion = ifCtx->else_expression() != nullptr;
      auto ifOp = builder.create<IfOp>(loc(*ifCtx->getStart()), conditional,
                                       elseRegion);
      builder.setInsertionPointToStart(ifOp.thenBlock());
      for (auto ifExpr : ifCtx->if_expression()->exprwbreak())
        if (failed(mlirGen(ifExpr)))
          return failure();
      if (elseRegion) {
        builder.setInsertionPointToStart(ifOp.elseBlock());
        for (auto elseExpr : ifCtx->else_expression()->exprwbreak())
          if (failed(mlirGen(elseExpr)))
            return failure();
      }
      builder.setInsertionPointAfter(ifOp);
      return success();
    }
    if (auto ifnotCtx = ctx->ifnot_stmt()) {
      auto conditional = mlirGen(ifnotCtx->cond_comb());
      // remember to negate the result
      conditional = builder.create<fsm::NegOp>(
          conditional.getLoc(), builder.getI1Type(), conditional);
      auto elseRegion = ifnotCtx->else_expression() != nullptr;
      auto ifOp = builder.create<IfOp>(loc(*ifnotCtx->getStart()), conditional,
                                       elseRegion);
      builder.setInsertionPointToStart(ifOp.thenBlock());
      for (auto ifExpr : ifnotCtx->if_expression()->exprwbreak())
        if (failed(mlirGen(ifExpr)))
          return failure();
      if (elseRegion) {
        builder.setInsertionPointToStart(ifOp.elseBlock());
        for (auto elseExpr : ifnotCtx->else_expression()->exprwbreak())
          if (failed(mlirGen(elseExpr)))
            return failure();
      }
      builder.setInsertionPointAfter(ifOp);
      return success();
    }
    return failure();
  }

  Value mlirGen(ProtoCCParser::Cond_combContext *ctx) {
    Value lhs = mlirGen(ctx->cond_rel(0));
    for (size_t i = 1; i < ctx->cond_rel().size(); i++) {
      auto oper = ctx->combinatorial_operator(i - 1)->getText();
      auto rhs = mlirGen(ctx->cond_rel(i));
      // TODO - implement AND/OR ops in FSM
      if (oper == "&")
        lhs = builder.create<AndOp>(loc(*ctx->getStart()), builder.getI1Type(),
                                    lhs, rhs);
      else
        lhs = builder.create<OrOp>(loc(*ctx->getStart()), builder.getI1Type(),
                                   lhs, rhs);
    }
    return lhs;
  }

  Value mlirGen(ProtoCCParser::Cond_relContext *ctx) {
    auto condSel = ctx->cond_sel();
    auto lhs = mlirGen(condSel->cond_type_expr(0));
    for (size_t i = 1; i < condSel->cond_type_expr().size(); i++) {
      auto oper = condSel->relational_operator(i - 1);
      auto rhs = mlirGen(condSel->cond_type_expr(i));
      lhs =
          builder.create<CompareOp>(loc(*ctx->getStart()), builder.getI1Type(),
                                    lhs, rhs, oper->getText());
    }
    return lhs;
  }

  Value mlirGen(ProtoCCParser::Cond_type_exprContext *ctx) {
    auto condType = ctx->cond_types(0);
    auto lhs = mlirGen(condType);
    for (size_t i = 1; i < ctx->cond_types().size(); i++) {
      auto rhsCtx = ctx->cond_types(i);
      auto indvMathOp = ctx->indv_math_op(i - 1);
      auto rhs = mlirGen(rhsCtx);
      if (indvMathOp->PLUS() != nullptr)
        lhs = builder.create<fsm::AddOp>(loc(*indvMathOp->PLUS()),
                                         builder.getI64Type(), lhs, rhs);
      else if (indvMathOp->MINUS())
        lhs = builder.create<fsm::SubOp>(loc(*indvMathOp->MINUS()),
                                         builder.getI64Type(), lhs, rhs);
      else
        assert(0 && "Multiply not yet supported!");
    }
    return lhs;
  }

  Value mlirGen(ProtoCCParser::Cond_typesContext *ctx) {
    if (auto objExprCtx = ctx->object_expr())
      return mlirGen(objExprCtx);
    if (auto setFn = ctx->set_func())
      return mlirGen(setFn);
    if (auto intCtx = ctx->INT())
      return builder.create<ConstOp>(
          loc(*intCtx), builder.getI64Type(),
          builder.getI64IntegerAttr(
              std::strtol(intCtx->getText().c_str(), nullptr, 10)));
    if (auto boolCtx = ctx->BOOL())
      return builder.create<ConstOp>(
          loc(*ctx->BOOL()), builder.getI1Type(),
          builder.getBoolAttr(boolCtx->getText() == "true"));

    if (auto nidCtx = ctx->NID())
      return builder.create<ReferenceOp>(loc(*nidCtx),
                                         IDType::get(builder.getContext()),
                                         builder.getSymbolRefAttr(curMach));

    assert(0 && "Invalid cond type expr");
  }

  LogicalResult mlirGen(ProtoCCParser::ExprwbreakContext *ctx) {
    // expression
    if (auto exprCtx = ctx->expressions())
      return mlirGen(exprCtx);

    // network send
    if (auto netSend = ctx->network_send())
      return mlirGen(netSend);

    // net mcast
    if (auto netMcast = ctx->network_mcast())
      return mlirGen(netMcast);

    // transaction
    if (auto transaction = ctx->transaction())
      return mlirGen(transaction);

    if (auto nextBreak = ctx->next_break()) {
      builder.create<BreakOp>(loc(*nextBreak->BREAK()));
      return success();
    }
    return failure();
  }

  LogicalResult mlirGen(ProtoCCParser::TransactionContext *ctx) {
    auto awaitOp = builder.create<AwaitOp>(loc(*ctx->AWAIT()));
    builder.setInsertionPointToStart(awaitOp.addEntryBlock());
    for (auto transCtx : ctx->trans()) {
      if (failed(mlirGen(transCtx)))
        return failure();
    }
    builder.setInsertionPointAfter(awaitOp);
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::TransContext *ctx) {
    auto msgName = ctx->ID();
    auto location = loc(*msgName);
    auto whenOp = builder.create<WhenOp>(
        location, msgName->getText(),
        builder.getFunctionType({MsgType::get(builder.getContext())}, {}));

    builder.setInsertionPointToStart(whenOp.addEntryBlock());

    assert(succeeded(declare(msgName->getText(), whenOp.getArgument(0))));

    for (auto tBodyCtx : ctx->trans_body())
      if (failed(mlirGen(tBodyCtx)))
        return failure();
    builder.setInsertionPointAfter(whenOp);
    return success();
  }

  LogicalResult mlirGen(ProtoCCParser::Trans_bodyContext *ctx) {
    // expression
    if (auto expr = ctx->expressions())
      return mlirGen(expr);
    // break
    if (auto nextBreakCtx = ctx->next_break())
      builder.create<BreakOp>(loc(*nextBreakCtx->BREAK()));
    // transaction
    if (auto transactionCtx = ctx->transaction())
      return mlirGen(transactionCtx);
    // net send
    if (auto netSendCtx = ctx->network_send())
      return mlirGen(netSendCtx);
    // net mcast
    if (auto netmcastCtx = ctx->network_mcast())
      return mlirGen(netmcastCtx);

    // TODO - support next_trans & net_bcast
    return success();
  }

  Value findVariableOp(const std::string &ident) {
    auto theMach = theModule.lookupSymbol<MachineOp>(curMach);
    for (auto var : theMach.body().getOps<VariableOp>())
      if (var.name() == ident)
        return var;
    return nullptr;
  }

  LogicalResult mlirGen(ProtoCCParser::AssignmentContext *ctx) {
    // assignment is only used in two cases
    // 1 = for updating the cache state cl=GetM_Ack_D.cl;
    // 2 - assigning to a local variable
    auto lhsIdent = ctx->process_finalident()->getText();
    // case 1 - update
    if (auto lhsValue = findVariableOp(lhsIdent)) {
      Type internalStateType = lhsValue.getType();
      Value rhs = mlirGen(ctx->assign_types());
      assert(rhs != nullptr);

      if (failed(areTypesCompatible(rhs.getType(), internalStateType)) &&
          rhs.getType() != internalStateType) {

        emitError(loc(*ctx->assign_types()->getStart()),
                  "Cannot assign Type '" + utils::getTypeString(rhs.getType()) +
                      "' to '" + utils::getTypeString(internalStateType) + "'");
        return failure();
      }
      builder.create<UpdateOp>(loc(*ctx->process_finalident()->getStart()),
                               lhsValue, rhs);

    }
    // Case 2 - local var
    else {
      return declare(lhsIdent, mlirGen(ctx->assign_types()));
    }
    return success();
  }

  Value mlirGen(ProtoCCParser::Assign_typesContext *ctx) {
    // INT
    if (auto intCtx = ctx->INT())
      return builder.create<ConstOp>(
          loc(*intCtx), builder.getI64Type(),
          builder.getI64IntegerAttr(
              std::strtol(intCtx->getText().c_str(), nullptr, 10)));
    // BOOL
    if (auto boolCtx = ctx->BOOL())
      return builder.create<ConstOp>(
          loc(*ctx->BOOL()), builder.getI1Type(),
          builder.getBoolAttr(boolCtx->getText() == "true"));

    // Obj_expr
    if (auto objExprCtx = ctx->object_expr())
      return mlirGen(objExprCtx);

    // message_constructor
    if (auto msgCtrCtx = ctx->message_constr())
      return mlirGen(msgCtrCtx);

    // math op
    if (auto mathOpCtx = ctx->math_op()) {
      auto lhs = mlirGen(mathOpCtx->val_range(0));
      auto rhs = mlirGen(mathOpCtx->val_range(1));
      if (mathOpCtx->PLUS()) {
        return builder.create<fsm::AddOp>(loc(*mathOpCtx->PLUS()),
                                          builder.getI64Type(), lhs, rhs);
      } else {
        return builder.create<fsm::SubOp>(loc(*mathOpCtx->PLUS()),
                                          builder.getI64Type(), lhs, rhs);
      }
    }

    // set_func
    if (auto setFnCtx = ctx->set_func())
      return mlirGen(setFnCtx);

    assert(0 && "Invalid Assign Types Value");
  }

  Value mlirGen(ProtoCCParser::Object_exprContext *ctx) {
    if (auto objIdCtx = ctx->object_id()) {
      auto ident = objIdCtx->getText();
      // If State
      if (stableStates.find(ident) != stableStates.end()) {
        return builder.create<ConstOp>(loc(*objIdCtx->ID()),
                                       StateType::get(builder.getContext()),
                                       builder.getStringAttr(ident));
      } else {
        return lookupWithSymbols(ident);
      }
    }

    if (auto objFnCtx = ctx->object_func()) {
      auto objId = objFnCtx->ID();
      Location location = loc(*objId);
      // if referencing directory
      if (objFnCtx->object_idres()->NID())
        return builder.create<ReferenceOp>(
            location, IDType::get(builder.getContext()),
            builder.getSymbolRefAttr(objId->getText()));

      auto object = lookupWithSymbols(objId->getText()).cast<BlockArgument>();
      assert(object != nullptr && "failed to find msg");
      auto address = objFnCtx->object_idres()->ID()->getText();
      assert(object.getType().isa<MsgType>() && "only msg types allowed");

      // get the type of the index
      Type resultingType;
      if (address == "src" ||
          address == "dst") // src & dst are fixed and always must be type ID
        resultingType = IDType::get(builder.getContext());
      else {
        auto parentOp = object.getOwner()->getParentOp();
        if (auto awaitPar = dyn_cast<WhenOp>(parentOp)) {
          auto declname = msgNameTypeMap.find(awaitPar.sym_name().str());
          assert(declname != msgNameTypeMap.end());
          auto msgDecl = theModule.lookupSymbol<MessageDecl>(declname->second);
          resultingType =
              msgDecl.lookupSymbol<MessageVariable>(address).getType();
        } else {
          auto procPar = dyn_cast<TransitionOp>(parentOp);
          auto declname = msgNameTypeMap.find(procPar.sym_name().str());
          assert(declname != msgNameTypeMap.end());
          auto msgDecl = theModule.lookupSymbol<MessageDecl>(declname->second);
          resultingType =
              msgDecl.lookupSymbol<MessageVariable>(address).getType();
        }
      }
      return builder.create<AccessOp>(location, resultingType, object, address);
    }
    assert(0 && "invalid obj expr");
  }

  Value mlirGen(ProtoCCParser::Message_constrContext *ctx) {
    Location msgLoc = loc(*ctx->ID());
    auto msgType = ctx->ID()->getText();
    auto msgName = ctx->message_expr(0)->getText();
    std::vector<Value> operands;
    for (size_t i = 1; i < ctx->message_expr().size(); i++) {
      operands.push_back(mlirGen(ctx->message_expr(i)));
    }

    //// Validate that we are constructing the message correctly
    // Lookup the request type
    auto msgTypeDecl = theModule.lookupSymbol<MessageDecl>(msgType);
    auto msgTypeVariablesIt =
        msgTypeDecl.body().front().getOps<MessageVariable>();
    std::vector<MessageVariable> msgTypeVariablesVec = {
        msgTypeVariablesIt.begin(), msgTypeVariablesIt.end()};

    /// Verify length arguments
    if (operands.size() != 2 + msgTypeVariablesVec.size()) {
      emitError(msgLoc, "Incorrect number of arguments in '" + msgType +
                            "' constructor : Expected '" +
                            std::to_string(msgTypeVariablesVec.size() + 2 + 1) +
                            "' got '" + std::to_string(operands.size() + 1) +
                            "'");
      return nullptr;
    }

    auto checkTypes = [](Type expected, Type actual, Location location,
                         const std::string &paramName) -> LogicalResult {
      if (failed(areTypesCompatible(expected, actual)) && actual != expected) {
        emitError(location, "Invalid type: Expected type '" +
                                utils::getTypeString(expected) +
                                "' for param '" + paramName + "' but got '" +
                                utils::getTypeString(actual) + "'");
        return failure();
      }
      return success();
    };
    /// Verify types of arguments
    for (size_t i = 0; i < operands.size(); i++) {
      auto operand = operands.at(i);
      auto operandType = operand.getType();
      auto paramLoc = loc(*ctx->message_expr(i + 1)->getStart());
      // dst and src are ID Types
      if (i < 2) {
        std::string paramName = i == 0 ? "src" : "dst";
        if(failed(checkTypes(IDType::get(builder.getContext()), operandType, paramLoc,
                   paramName)))
          return nullptr;
      } else {
        auto expectedType = msgTypeVariablesVec.at(i - 2).getType();
        auto paramName = msgTypeVariablesVec.at(i - 2).sym_name().str();
        if(failed(checkTypes(expectedType, operandType, paramLoc, paramName)))
          return nullptr;
      }
    }
    return builder.create<MessageOp>(msgLoc, MsgType::get(builder.getContext()),
                                     builder.getSymbolRefAttr(msgType),
                                     builder.getStringAttr(msgName), operands);
  }

  Value mlirGen(ProtoCCParser::Message_exprContext *ctx) {
    // obj_expr
    if (auto objExprCtx = ctx->object_expr())
      return mlirGen(objExprCtx);
    // set_fn
    if (auto setFnCtx = ctx->set_func())
      return mlirGen(setFnCtx);

    // NID
    if (auto nidCtx = ctx->NID())
      return builder.create<ReferenceOp>(loc(*nidCtx),
                                         IDType::get(builder.getContext()),
                                         builder.getSymbolRefAttr(curMach));

    // INT
    if (auto intCtx = ctx->INT())
      return builder.create<ConstOp>(
          loc(*intCtx), builder.getI64Type(),
          builder.getI64IntegerAttr(
              std::strtol(intCtx->getText().c_str(), nullptr, 10)));

    // BOOL
    if (auto boolCtx = ctx->BOOL())
      return builder.create<ConstOp>(
          loc(*boolCtx), builder.getI1Type(),
          builder.getBoolAttr(
              std::strtol(boolCtx->getText().c_str(), nullptr, 10)));
    assert(0 && "Invalid Message Expression");
  }

  Value mlirGen(ProtoCCParser::Set_funcContext *ctx) {
    auto setId = ctx->ID()->getText();
    Location location = loc(*ctx->ID());
    Value theSet = lookupWithSymbols(setId);
    auto fnType = ctx->set_function_types()->getText();

    if (fnType == "add") {
      builder.create<SetAdd>(location, theSet, mlirGen(ctx->set_nest(0)));
      return nullptr;
    }
    if (fnType == "count")
      return builder.create<SetCount>(location, builder.getI64Type(), theSet);

    if (fnType == "contains")
      return builder.create<SetContains>(location, builder.getI1Type(), theSet,
                                         mlirGen(ctx->set_nest(0)));

    if (fnType == "del") {
      builder.create<SetDelete>(location, theSet, mlirGen(ctx->set_nest(0)));
      return nullptr;
    }

    if (fnType == "clear") {
      builder.create<SetClear>(location, theSet);
      return nullptr;
    }
    assert(0 && "invalid set fn");
  }

  Value mlirGen(ProtoCCParser::Set_nestContext *ctx) {
    if (auto setFnCtx = ctx->set_func())
      return mlirGen(setFnCtx);
    if (auto objExprCtx = ctx->object_expr())
      return mlirGen(objExprCtx);
    assert(0 && "invalid");
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