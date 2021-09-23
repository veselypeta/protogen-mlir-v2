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
#include <set>
#include <utility>

namespace {

/*
 * Useful constants
 */
constexpr auto mtype_a = "mtype";
constexpr auto default_mtype_t = "none";
constexpr auto src_a = "src";
constexpr auto dst_a = "dst";
constexpr auto id_a = "id";

class MLIRGenImpl {

public:
  explicit MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
  mlir::ModuleOp mlirGen(ProtoCCParser::DocumentContext *ctx,
                         std::string compFile) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    // save the filename - used for location tracking
    filename = std::move(compFile);

    // set the insertion point to the start of the module
    builder.setInsertionPointToStart(theModule.getBody());

    // declare a global scope
    SymbolTableScopeT global_scope(symbolTable);

    // recursively call mlirGen
    if (mlir::failed(mlirGen(ctx))) {
      assert(0 && "failed to generate MLIR correctly!");
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

  /*
   * Message Inheritance
   *
   * All messages constructed in PCC inherit from the base Message type
   * which has the following three fields
   *
   * mtype : a string-like identifier to the Message Type
   * src : an identifier for the machine from which the message has been sent
   * dst : an identifier for the machine from which the message is to be sent
   */
  // A type map for mapping string -> mlir::Type
  using TypeMapT = std::map<std::string, mlir::Type>;
  TypeMapT BaseMsg = {
      std::make_pair(mtype_a, mlir::pcc::MsgIdType::get(builder.getContext(),
                                                        default_mtype_t)),
      {src_a, mlir::pcc::IDType::get(builder.getContext())},
      {dst_a, mlir::pcc::IDType::get(builder.getContext())}};

  // used to hold underlying data for llvm::StringRef
  std::set<std::string> identStorage;

  // A symbol table is used to hold (ident -> mlir::Value) references
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

  // return an mlir::Location object for builder operations
  mlir::Location loc(const antlr4::Token &tok) const {
    return mlir::FileLineColLoc::get(filename, tok.getLine(),
                                     tok.getCharPositionInLine(),
                                     builder.getContext());
  }

  mlir::Location loc(antlr4::tree::TerminalNode &terminal) const {
    return loc(*terminal.getSymbol());
  }

  // Used to declare MLIR Values along with their identifiers
  mlir::LogicalResult declare(std::string &ident, mlir::Value val) {
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

  mlir::Value lookup(std::string &ident) {
    // Identifier must already be in the symbol table - otherwise error!
    if (!symbolTable.count(ident)) {
      assert(0 && "attempting to lookup ident which is not declared!");
    }
    return symbolTable.lookup(ident);
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::DocumentContext *ctx) {
    // recursively call mlirGen on const_decl operations
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

    // recursively call mlirGen on arch_block
    for (auto archBlockCtx : ctx->arch_block()) {
      if (archBlockCtx) {
        if (mlir::failed(mlirGen(archBlockCtx)))
          return mlir::failure();
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Const_declContext *ctx) {
    std::string constId = ctx->ID()->getText();
    std::string intText = ctx->INT()->getText();
    long constValue = strtol(ctx->INT()->getText().c_str(), nullptr, 10);
    mlir::Location constDeclLocation = loc(*ctx->ID());

    auto constOp = builder.create<mlir::pcc::ConstantOp>(constDeclLocation,
                                                         constId, constValue);

    return declare(constId, constOp);
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Init_hwContext *ctx) {
    // recursively call for each network block
    if (mlir::failed(mlirGen(ctx->network_block())))
      return mlir::failure();
    // recursively call for each message block
    if (mlir::failed(mlirGen(ctx->message_block())))
      return mlir::failure();
    // recursively call for each machines block
    if (mlir::failed(mlirGen(ctx->machines())))
      return mlir::failure();

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Network_blockContext *ctx) {
    // skip if the recursive call is nullptr
    if (ctx == nullptr)
      return mlir::success();
    for (auto network : ctx->network_element()) {
      std::string networkId = network->ID()->getText();
      mlir::Location netLoc = loc(*network->ID());
      std::string orderingStr = network->element_type()->getText();
      // lowercase the string
      std::transform(orderingStr.begin(), orderingStr.end(),
                     orderingStr.begin(), ::tolower);
      // construct the appropriate type - ordered/unordered
      mlir::pcc::NetworkType netType = mlir::pcc::NetworkType::get(
          builder.getContext(),
          mlir::pcc::NetworkType::convertToOrder(orderingStr));

      auto idAttr = builder.getStringAttr(networkId);
      auto netOp =
          builder.create<mlir::pcc::NetDeclOp>(netLoc, netType, idAttr);
      // Declare the network op in the scope
      if (failed(declare(networkId, netOp)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Message_blockContext *ctx) {
    // skip if nullptr
    if (ctx == nullptr)
      return mlir::success();

    // get necessary info for operation construction
    std::string msgId = ctx->ID()->getText(); // i.e. Resp
    mlir::Location msgLoc = loc(*ctx->ID());

    // initialize arrays for operation construction
    std::vector<mlir::NamedAttribute> field_attrs;
    std::vector<mlir::Type> struct_types;
    field_attrs.reserve(ctx->declarations().size());
    struct_types.reserve(ctx->declarations().size());

    // initialize a local msg type map for individual MsgTypes
    // i.e. Resp
    TypeMapT localMsgTypeMap = BaseMsg; // copy the BaseMsg

    auto add_to_declarations =
        [&](std::pair<std::string, mlir::Type> &&declPair) {
          struct_types.emplace_back(declPair.second);
          field_attrs.emplace_back(std::make_pair(
              mlir::Identifier::get(declPair.first, builder.getContext()),
              mlir::TypeAttr::get(declPair.second)));
        };

    std::for_each(localMsgTypeMap.begin(), localMsgTypeMap.end(),
                  add_to_declarations);

    for (auto msgDecl : ctx->declarations()) {
      add_to_declarations(mlirTypeGen(msgDecl));
    }

    mlir::pcc::StructType msg_struct_type =
        mlir::pcc::StructType::get(struct_types);
    mlir::pcc::MsgDeclOp msg_decl = builder.create<mlir::pcc::MsgDeclOp>(
        msgLoc, msgId, msg_struct_type, field_attrs);
    return declare(msgId, msg_decl);
  }

  std::pair<std::string, mlir::Type>
  mlirTypeGen(ProtoCCParser::DeclarationsContext *ctx) {
    if (ctx->int_decl())
      return mlirTypeGen(ctx->int_decl());
    if (ctx->bool_decl())
      return mlirTypeGen(ctx->bool_decl());
    if (ctx->state_decl())
      return mlirTypeGen(ctx->state_decl());
    if (ctx->id_decl())
      return mlirTypeGen(ctx->id_decl());
    if (ctx->data_decl())
      return mlirTypeGen(ctx->data_decl());
    assert(0 && "declaration did not match any of the supported types");
  }

  std::pair<std::string, mlir::pcc::IntRangeType>
  mlirTypeGen(ProtoCCParser::Int_declContext *ctx) {
    // Integer declarations always define an integer range
    // we find the start and stop of the integer range
    std::string intDeclId = ctx->ID()->getText();

    // get a reference to the start and end sub-range decls
    auto startRange = ctx->range()->val_range().at(0);
    auto endRange = ctx->range()->val_range().at(1);
    // will always be 2 long due to grammar rules

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
      return strtol(ctx->INT()->getText().c_str(), nullptr, 10);

    // if ID we need to lookup in symbol table
    std::string constRef = ctx->ID()->getText();
    mlir::Value valRef = lookup(constRef);
    // we know this must have come from a ConstantOp
    auto constOp = static_cast<mlir::pcc::ConstantOp>(valRef.getDefiningOp());
    return constOp.val();
  }

  // Mlir generate for machines
  mlir::LogicalResult mlirGen(ProtoCCParser::MachinesContext *ctx) {
    // skip if nullptr
    if (ctx == nullptr)
      return mlir::success();
    // generate typedef for caches
    if (ctx->cache_block()) {
      return mlirGen(ctx->cache_block());
    }
    if (ctx->dir_block()) {
      return mlirGen(ctx->dir_block());
    }
    // fail if not cache or directory
    assert(0 &&
           "machine types other than (cache or directory) are not supported!");
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Cache_blockContext *ctx) {
    std::string cache_id = ctx->ID()->getText();
    mlir::Location cache_loc = loc(*ctx->ID());

    std::vector<mlir::NamedAttribute> field_attributes;
    field_attributes.reserve(ctx->declarations().size());
    std::vector<mlir::Type> struct_types;
    struct_types.reserve(ctx->declarations().size());

    for (auto cache_decl : ctx->declarations()) {
      auto declPair = mlirTypeGen(cache_decl);
      assert(declPair.first != id_a && "field 'id' is a reserved identifier, "
                                       "do not use in Cache definition");
      // add the type pair to the named attributes
      field_attributes.emplace_back(std::make_pair(
          mlir::Identifier::get(declPair.first, builder.getContext()),
          mlir::TypeAttr::get(declPair.second)));
      struct_types.emplace_back(declPair.second);
    }
    // construct the cache struct type
    mlir::pcc::StructType cache_struct =
        mlir::pcc::StructType::get(struct_types);

    // if a set of caches
    if (ctx->objset_decl() != nullptr) {
      auto set_size = getIntFromValRange(ctx->objset_decl()->val_range());
      mlir::pcc::SetType set_type =
          mlir::pcc::SetType::get(cache_struct, set_size);
      mlir::pcc::CacheDeclOp cache_decl =
          builder.create<mlir::pcc::CacheDeclOp>(cache_loc, cache_id, set_type,
                                                 field_attributes);
      return declare(cache_id, cache_decl);
    }
    mlir::pcc::CacheDeclOp cache_decl = builder.create<mlir::pcc::CacheDeclOp>(
        cache_loc, cache_id, cache_struct, field_attributes);
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Dir_blockContext *ctx) {
    std::string dir_id = ctx->ID()->getText();
    mlir::Location dir_loc = loc(*ctx->ID());

    std::vector<mlir::NamedAttribute> field_attrs;
    std::vector<mlir::Type> struct_types;
    field_attrs.reserve(ctx->declarations().size());
    struct_types.reserve(ctx->declarations().size());

    for (auto decl : ctx->declarations()) {
      auto declPair = mlirTypeGen(decl);
      assert(declPair.first != id_a && "field 'id' is a reserved identifier, "
                                       "do not use in Directory definition");
      struct_types.emplace_back(declPair.second);
      field_attrs.emplace_back(std::make_pair(
          mlir::Identifier::get(declPair.first, builder.getContext()),
          mlir::TypeAttr::get(declPair.second)));
    }
    auto x = ctx->objset_decl();
    assert(ctx->objset_decl().empty() &&
           "sets in directory are not supported!");
    mlir::pcc::StructType dir_struct = mlir::pcc::StructType::get(struct_types);
    mlir::pcc::DirectoryDeclOp dir_decl_op =
        builder.create<mlir::pcc::DirectoryDeclOp>(dir_loc, dir_id, dir_struct,
                                                   field_attrs);
    return declare(dir_id, dir_decl_op);
  }

  // MLIR generate for architecture blocks
  mlir::LogicalResult mlirGen(ProtoCCParser::Arch_blockContext *ctx) {
    // skip if nullptr
    if (!ctx)
      return mlir::success();

    //    // Lookup the Arch we are targeting from the ID
    //    std::string archId = ctx->ID()->getText();
    //    mlir::Value archSSA = lookup(archId);
    //
    //    // Stable State Definitions ie. {M, S, I};
    //    std::vector<std::string> stableStates;
    //    ProtoCCParser::Stable_defContext *stableCtx =
    //        ctx->arch_body()->stable_def();
    //    for (auto stableState : stableCtx->ID()) {
    //      stableStates.push_back(stableState->getText());
    //    }
    //
    //    // for each process block
    //    for (auto procBlockCtx : ctx->arch_body()->process_block()) {
    //      // Get the Start State of the transaction
    //      ProtoCCParser::Process_transContext *transContext =
    //          procBlockCtx->process_trans();
    //      std::string startState = transContext->ID()->getText();
    //
    //      // TODO -- we are just getting the string value of the action
    //      // Better would be to discriminate between (ACCESS, EVICT, ID)
    //      std::string action = transContext->process_events()->getText();
    //
    //      // get the final state if it exists
    //      std::string finalState;
    //      if (transContext->process_finalstate())
    //        finalState =
    //            transContext->process_finalstate()->process_finalident()->getText();
    //
    //      // TODO - create an operation that takes the correct types as
    //      parameters auto archType = archSSA.getType(); auto msgType =
    //      getGlobalMsgType(); llvm::SmallVector<mlir::Type, 2> procArgs;
    //      procArgs.push_back(archType);
    //      procArgs.push_back(msgType);
    //      auto procType = builder.getFunctionType(procArgs, llvm::None);
    //
    //      std::string processIdent = (startState += "_") += action;
    //
    //      auto procOpLocation = loc(*procBlockCtx->PROC());
    //      auto procOp = builder.create<mlir::pcc::ProcessOp>(
    //          procOpLocation, processIdent, procType);
    //      auto &entryBlock = *procOp.addEntryBlock();
    //      builder.setInsertionPointToStart(&entryBlock);
    //
    //      // TODO -- insert Operations into the Process Operation
    //      for (auto expr : procBlockCtx->process_expr()) {
    //        // Generate Expressions
    //        if (expr->expressions()) {
    //          if (mlir::failed(mlirGen(expr->expressions()))) {
    //            return mlir::failure();
    //          }
    //        }
    //      }
    //
    //      builder.create<mlir::pcc::BreakOp>(builder.getUnknownLoc());
    //      // reset the insertion point to after the Process op
    //      builder.setInsertionPointAfter(procOp);
    //    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::ExpressionsContext *ctx) {
    if (ctx->assignment()) {
      // get the assingment ident i.e. msg
      std::string asignmentId =
          ctx->assignment()->process_finalident()->getText();
      auto assignTypesCtx = ctx->assignment()->assign_types();
      // inline for now
      //      if (assignTypesCtx->message_constr()) {
      //        // get the type of msg being constructed i.e. Request
      //        std::string msgTypeId =
      //            assignTypesCtx->message_constr()->ID()->getText();
      //        // find the type of the message we wish to construct
      //        auto msgConstrTypeMap = getMsgMap(msgTypeId);
      //        auto msgConstrParams =
      //        assignTypesCtx->message_constr()->message_expr();
      //        // TODO -- this can be uncommented once we push the name
      //        parameter to
      //        // msg type assert(msgConstrTypeMap.size() ==
      //        msgConstrParams.size() &&
      //        // "parameter length should match!");
      //        auto fields = getFieldIdsFromMap(msgConstrTypeMap);
      //        for (size_t i = 0; i < msgConstrTypeMap.size(); i++) {
      //          auto paramCtx = msgConstrParams[i];
      //          mlirGen(paramCtx);
      //        }
      //      }
    }
    return mlir::success();
  }

  void mlirGen(ProtoCCParser::Message_exprContext *ctx) {
    if (ctx->INT()) {
      buildIntConstant(*ctx->INT());
    }
    if (ctx->BOOL()) {
      buildBoolConstant(*ctx->BOOL());
    }
    if (ctx->NID()) {
      //      buildSelfIdReference(*ctx->NID());
    }
  }

  mlir::Value buildIntConstant(antlr4::tree::TerminalNode &intTok) {

    auto intVal = std::stoi(intTok.getText());
    return builder.create<mlir::ConstantOp>(loc(intTok),
                                            builder.getI64IntegerAttr(intVal));
  }

  mlir::Value buildBoolConstant(antlr4::tree::TerminalNode &boolTok) {
    bool boolVal = boolTok.getText() == "true";
    return builder.create<mlir::ConstantOp>(loc(boolTok),
                                            builder.getBoolAttr(boolVal));
  }

  mlir::Value buildSelfIdReference(antlr4::tree::TerminalNode &nid) {
    return builder.create<mlir::pcc::FooOp>(loc(nid), builder.getI64Type());
  }
};
} // namespace

namespace pcc {
mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx,
                       ProtoCCParser::DocumentContext *docCtx,
                       std::string filename) {
  return MLIRGenImpl(mlirCtx).mlirGen(docCtx, std::move(filename));
}
} // namespace pcc
