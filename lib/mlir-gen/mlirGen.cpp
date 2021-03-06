#include "mlir-gen/mlirGen.h"
#include "PCC/PCCOps.h"
#include "ProtoCCBaseVisitor.h"

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

namespace mlirGenImpl {

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

    // preprocess message names
    preprocessMsgNames(ctx);

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

  std::string curMach; // tells us which machine we are currently parsing for
  std::vector<std::string> stable_states;

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

  // It's useful to hold a map for struct like cache/directory
  std::map<std::string, TypeMapT> machStructMap;

  // When preprocessing PCC: we can deduce each pairing of MsgName->MsgType i.e.
  // GetM->Request
  std::map<std::string, std::string> msgNameToMsgTypeMap;

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
    msgNameToMsgTypeMap = msgVisitor.msgNameMap;
  }

  std::string resolveMsgName(const std::string &id) {
    auto find = msgNameToMsgTypeMap.find(id);
    assert(find != msgNameToMsgTypeMap.end() &&
           "could not a mapping for message name to a specific type");
    return find->second;
  }

  // return an mlir::Location object for builder operations
  mlir::Location loc(const antlr4::Token &tok) const {
    return mlir::FileLineColLoc::get(
        mlir::Identifier::get(filename, builder.getContext()), tok.getLine(),
        tok.getCharPositionInLine());
  }

  mlir::Location loc(antlr4::tree::TerminalNode &terminal) const {
    return loc(*terminal.getSymbol());
  }

  // Used to declare MLIR Values along with their identifiers
  mlir::LogicalResult declare(const std::string &ident, mlir::Value val) {
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

  void registerStructType(llvm::StringRef structId,
                          const std::vector<mlir::NamedAttribute> &attrs) {
    TypeMapT typeMap;
    for (auto attr : attrs) {
      std::string fieldId = attr.first.str();
      mlir::Type type = attr.second.cast<mlir::TypeAttr>().getValue();
      typeMap.insert({fieldId, type});
    }
    machStructMap.insert({structId.str(), typeMap});
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

    // register the struct in the global struct map
    registerStructType(cache_id, field_attributes);

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
    mlir::pcc::CacheDeclOp cache_decl __attribute__((unused)) = builder.create<mlir::pcc::CacheDeclOp>(
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

    // register the struct
    registerStructType(dir_id, field_attrs);
    return declare(dir_id, dir_decl_op);
  }

  // MLIR generate for architecture blocks
  mlir::LogicalResult mlirGen(ProtoCCParser::Arch_blockContext *ctx) {
    // skip if nullptr
    if (!ctx)
      return mlir::success();

    // Lookup the Arch we are targeting from the ID
    std::string archId = ctx->ID()->getText();
    mlir::Value archSSA = lookup(archId);
    curMach = archId;

    // Stable State Definitions ie. {M, S, I};
    std::vector<std::string> stableStates;
    ProtoCCParser::Stable_defContext *stableCtx =
        ctx->arch_body()->stable_def();
    for (auto stableState : stableCtx->ID()) {
      stableStates.push_back(stableState->getText());
    }
    stable_states = stableStates;

    for (auto procBlockCtx : ctx->arch_body()->process_block()) {
      if (!mlir::succeeded(mlirGen(procBlockCtx, archSSA))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  struct ProcessTransResponse {
    enum class ActionKind { ID, EVICT, ACCESS };
    std::string start_state;
    std::string action;
    ActionKind action_kind;
    llvm::Optional<std::string> end_state;
  };

  static ProcessTransResponse
  parseProcessTransCtx(ProtoCCParser::Process_transContext *ctx) {
    std::string start_state = ctx->ID()->getText();
    std::string action = ctx->process_events()->getText();
    ProcessTransResponse::ActionKind action_kind;
    if (ctx->process_events()->ACCESS())
      action_kind = ProcessTransResponse::ActionKind::ACCESS;
    if (ctx->process_events()->EVICT())
      action_kind = ProcessTransResponse::ActionKind::EVICT;
    if (ctx->process_events()->ID())
      action_kind = ProcessTransResponse::ActionKind::ID;
    llvm::Optional<std::string> end_state;
    if (ctx->process_finalstate() &&
        !ctx->process_finalstate()->process_finalident()->isEmpty()) {
      end_state = ctx->process_finalstate()->process_finalident()->getText();
    }
    return {start_state, action, action_kind, end_state};
  }

  std::string getArgIdent(const std::string &id) const { return "arg" + id; }

  mlir::LogicalResult mlirGen(ProtoCCParser::Process_blockContext *ctx,
                              mlir::Value archSSA) {
    if (!ctx)
      return mlir::success();

    SymbolTableScopeT process_scope(symbolTable);
    auto transContext = ctx->process_trans();
    auto transResp = parseProcessTransCtx(transContext);

    mlir::Location processOpLoc = loc(*transContext->ID());

    // Setup the parameter types
    llvm::SmallVector<std::string, 2> argNames;
    llvm::SmallVector<mlir::Type, 2> inputTypes;
    if (archSSA.getType().isa<mlir::pcc::SetType>()) {
      auto setType = archSSA.getType().cast<mlir::pcc::SetType>();
      assert(setType.getElementType().isa<mlir::pcc::StructType>() &&
             "set type did not have struct as inner type");
      inputTypes.push_back(setType.getElementType());
    } else {
      assert(archSSA.getType().isa<mlir::pcc::StructType>() &&
             "directory must be struct");
      inputTypes.push_back(archSSA.getType());
    }
    argNames.push_back(getArgIdent(curMach));

    // if msg type - then process has additional parameter the msg
    if (transResp.action_kind == ProcessTransResponse::ActionKind::ID) {
      std::string msgTypeIdent = resolveMsgName(transResp.action);
      mlir::Type actualType = lookup(msgTypeIdent).getType();
      inputTypes.push_back(actualType);
      argNames.push_back(getArgIdent(transResp.action));
    }

    auto procType = builder.getFunctionType(inputTypes, llvm::None);

    // Which attributes might be useful for performing optimizations ??
    // {action} store the name of the action taking place i.e. load/GetM
    mlir::Attribute actionAttr = builder.getStringAttr(transResp.action);
    // {start_state} have a reference to the start state i.e.
    // cache_I/directory_M
    mlir::Attribute startStateAttr =
        builder.getStringAttr(curMach + "_" + transResp.start_state);

    // {state_type} will be either 'stable' or 'transient'
    // initially all are stable by definition of SSP
    mlir::Attribute stateTypeAttr = builder.getStringAttr("stable");

    std::vector<mlir::NamedAttribute> procAttrs{
        {mlir::Identifier::get("action", builder.getContext()), actionAttr},
        {mlir::Identifier::get("start_state", builder.getContext()),
         startStateAttr},
        {mlir::Identifier::get("state_type", builder.getContext()),
         stateTypeAttr}};

    // {end_state} optional
    if (transResp.end_state.hasValue() &&
        transResp.end_state.getValue() != "State") {
      mlir::Attribute endStateAttr =
          builder.getStringAttr(transResp.end_state.getValue());
      procAttrs.emplace_back(std::make_pair(
          mlir::Identifier::get("end_state", builder.getContext()),
          endStateAttr));
    }
    std::string procIdent =
        curMach + "_" + transResp.start_state + "_" + transResp.action;
    auto procOp = builder.create<mlir::pcc::ProcessOp>(processOpLoc, procIdent,
                                                       procType, procAttrs);
    // TODO - come back to implementing parameters for the function op
    auto entryBlock = procOp.addEntryBlock();

    // declare the parameters of the function body:
    for (const auto nameValue :
         llvm::zip(argNames, entryBlock->getArguments())) {
      if (mlir::failed(declare(std::get<0>(nameValue), std::get<1>(nameValue))))
        return mlir::failure();
    }
    builder.setInsertionPointToStart(entryBlock);

    if (mlir::failed(mlirGen(ctx->process_expr())))
      return mlir::failure();

    builder.setInsertionPointAfter(procOp);
    return mlir::success();
  }

  mlir::LogicalResult
  mlirGen(const std::vector<ProtoCCParser::Process_exprContext *> &exprs) {
    if (exprs.empty())
      return mlir::success();
    for (auto *exprCtx : exprs) {
      if (mlir::failed(mlirGen(exprCtx)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Process_exprContext *ctx) {
    if (!ctx)
      return mlir::failure();

    if (ctx->expressions())
      return mlirGen(ctx->expressions());

    if (ctx->transaction()) {
      return mlirGen(ctx->transaction());
    }

    if (ctx->network_send()) {
      return mlirGen(ctx->network_send());
    }

    assert(0 && "Trying to parse expression types that are not supported");
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::ExpressionsContext *ctx) {
    if (!ctx)
      return mlir::success();

    if (ctx->assignment())
      return mlirGen(ctx->assignment());

    if (ctx->conditional())
      return mlirGen(ctx->conditional());

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::AssignmentContext *ctx) {
    std::string assignmentId = ctx->process_finalident()->getText();
    mlir::Location location = loc(*ctx->assign_types()->getStart());
    // i.e. State=S or cl=GetM.cl etc.
    auto check_if_ass_refs_mach_state = [&](llvm::StringRef ass_id) {
      // is State (keyword)
      return ctx->process_finalident()->STATE() ||
             // is part of the machine
             lookup(curMach).getDefiningOp()->hasAttr(ass_id);
    };
    if (check_if_ass_refs_mach_state(assignmentId)) {
      mlir::Value rv = mlirGen(ctx->assign_types(), true);
      // get the machine reference
      mlir::Value machParm = lookup(getArgIdent(curMach));
      builder.create<mlir::pcc::StateUpdateOp>(
          location, machParm, rv, builder.getStringAttr(assignmentId));
      return mlir::success();
    } else {
      return declare(assignmentId, mlirGen(ctx->assign_types()));
    }
  }

  mlir::Value mlirGen(ProtoCCParser::Assign_typesContext *ctx,
                      bool stateUpdate = false) {
    if (ctx->message_constr())
      return mlirGen(ctx->message_constr());
    if (ctx->object_expr() && stateUpdate) {
      return mlirGen(ctx->object_expr());
    }
    assert(0 && "currently unimplemented assignment types");
  }

  mlir::Value mlirGen(ProtoCCParser::Message_constrContext *ctx) {
    std::string msgTypeId = ctx->ID()->getText();
    mlir::Location msgLoc = loc(*ctx->ID());
    // lookup the type
    mlir::Value msgTypeConstr __attribute__((unused)) = lookup(msgTypeId);
    mlir::pcc::MsgDeclOp msgDeclOp =
        mlir::dyn_cast<mlir::pcc::MsgDeclOp>(lookup(msgTypeId).getDefiningOp());

    assert(msgDeclOp->getAttrs().size() - 1 == ctx->message_expr().size() &&
           "sizes match");
    auto msg_id_node = ctx->message_expr().at(0);
    auto srcVal = mlirGen(ctx->message_expr().at(1));
    auto dstVal = mlirGen(ctx->message_expr().at(2));

    mlir::pcc::MsgIdType msgType =
        mlir::pcc::MsgIdType::get(builder.getContext(), msg_id_node->getText());

    mlir::TypeAttr msgTypeAttr = mlir::TypeAttr::get(msgType);
    return builder.create<mlir::pcc::MsgConstrOp>(msgLoc, msgDeclOp.getType(),
                                                  srcVal, dstVal, msgTypeAttr);
  }

  mlir::Value mlirGen(ProtoCCParser::Message_exprContext *ctx) {
    if (ctx->INT()) {
      return buildIntConstant(*ctx->INT());
    }
    if (ctx->BOOL()) {
      return buildBoolConstant(*ctx->BOOL());
    }
    if (ctx->NID()) {
      return buildSelfIdReference(*ctx->NID());
    }
    if (ctx->object_expr()) {
      return mlirGen(ctx->object_expr());
    }
    assert(0 && "fell of the end of message expression");
  }

  mlir::Value mlirGen(ProtoCCParser::Object_exprContext *ctx) {
    if (ctx->object_func())
      return mlirGen(ctx->object_func());

    if (ctx->object_id())
      return mlirGen(ctx->object_id());

    assert(0 && "ran off the end of expressions context");
  }

  mlir::Value mlirGen(ProtoCCParser::Object_funcContext *ctx) {
    auto location = loc(*ctx->ID());
    // HERE - we only allow reference to single object, and attribute
    // i.e. directory.id or GetM.src

    std::string rootObject = ctx->ID()->getText();
    std::string index = ctx->object_idres()->getText();

    auto is_obj_mach = [&](llvm::StringRef /*ident*/) {
      return symbolTable.count(rootObject) > 0;
    };

    // if the value can be looked up
    if (is_obj_mach(rootObject)) {
      auto typeMap = machStructMap.at(rootObject);
      auto resultType = ctx->object_idres()->NID()
                            ? mlir::pcc::IDType::get(builder.getContext())
                            : typeMap.at(index);
      return builder.create<mlir::pcc::StructAccessOp>(
          location, resultType, lookup(rootObject),
          builder.getStringAttr(index));
    } else {

      mlir::Value inMsgArg = lookup(getArgIdent(rootObject));
      std::string msgType = resolveMsgName(rootObject);
      // lookup the msg declaration
      mlir::pcc::MsgDeclOp msgDecl =
          lookup(msgType).getDefiningOp<mlir::pcc::MsgDeclOp>();
      // get the type attr of the index
      mlir::Type resultType = msgDecl.getFieldAttr(index).getValue();
      return builder.create<mlir::pcc::StructAccessOp>(
          location, resultType, inMsgArg, builder.getStringAttr(index));
    }
  }

  mlir::Value mlirGen(ProtoCCParser::Object_idContext *ctx) {
    std::string reference = ctx->getText();
    mlir::Location location = loc(*ctx->ID());
    auto is_in_mach_map = [&](llvm::StringRef ref) {
      auto mapEntry = machStructMap.at(curMach);
      return mapEntry.find(ref.str()) != mapEntry.end();
    };
    auto is_local_variable = [&](llvm::StringRef ref) {
      return symbolTable.count(ref);
    };
    auto is_stable_state = [&](llvm::StringRef ref) {
      return std::find(stable_states.begin(), stable_states.end(), ref) !=
             stable_states.end();
    };

    // it is either a local variable
    if (is_local_variable(reference))
      return lookup(reference);
    // or the ident can be referencing a value in the machines state i.e. cl or
    // acksExpected etc.
    else if (is_in_mach_map(reference)) {
      auto mapEntry = machStructMap.at(curMach);
      mlir::Type resultType = mapEntry.at(reference);
      return builder.create<mlir::pcc::StructAccessOp>(
          location, resultType, lookup(getArgIdent(curMach)),
          builder.getStringAttr(reference));
    }
    // referring to a stable state
    else if (is_stable_state(reference)) {
      return builder.create<mlir::pcc::InlineConstOp>(
          location, mlir::pcc::StateAttr::get(builder.getContext(), reference));
    }
    assert(0 && "Invalid reference");
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
    mlir::Location paramLoc = loc(nid);

    return builder.create<mlir::pcc::StructAccessOp>(
        paramLoc, mlir::pcc::IDType::get(builder.getContext()),
        builder.getBlock()->getArgument(0),
        builder.getStringAttr(nid.getText()));
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Network_sendContext *ctx) {
    std::string netId = ctx->ID().at(0)->getText();
    std::string msgIdent = ctx->ID().at(1)->getText();
    mlir::Location sendLoc = loc(*ctx->send_function()->getStart());
    mlir::Value netSSA = lookup(netId);
    mlir::Value msgSSA = lookup(msgIdent);
    builder.create<mlir::pcc::MsgSendOp>(sendLoc, netSSA, msgSSA);
    return mlir::success();
  }

  /// MLIR GEN FOR TRANSACTIONS ///
  mlir::LogicalResult mlirGen(ProtoCCParser::TransactionContext *ctx) {
    mlir::Location transLoc = loc(*ctx->AWAIT());
    mlir::pcc::TransactionOp transOp =
        builder.create<mlir::pcc::TransactionOp>(transLoc);
    mlir::Block *entry = transOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    if (mlir::failed(mlirGen(ctx->trans())))
      return mlir::failure();

    builder.setInsertionPointAfter(transOp);

    return mlir::success();
  }

  mlir::LogicalResult
  mlirGen(const std::vector<ProtoCCParser::TransContext *> &ctx) {
    for (auto *transCtx : ctx) {
      if (mlir::failed(mlirGen(transCtx)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::TransContext *ctx) {
    std::string msgName = ctx->ID()->getText();
    mlir::Location transLoc = loc(*ctx->WHEN());
    const std::string msgTypeName = resolveMsgName(msgName);
    mlir::pcc::PCCType msgType =
        lookup(msgTypeName).getType().cast<mlir::pcc::StructType>();

    mlir::pcc::TransitionOp transOp = builder.create<mlir::pcc::TransitionOp>(
        transLoc, builder.getStringAttr(msgName), mlir::TypeAttr::get(msgType));

    mlir::Block *entry = transOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    if (mlir::failed(declare(getArgIdent(msgName), entry->getArgument(0))))
      return mlir::failure();

    if (mlir::failed(mlirGen(ctx->trans_body())))
      return mlir::failure();

    builder.setInsertionPointAfter(transOp);

    return mlir::success();
  }

  mlir::LogicalResult
  mlirGen(const std::vector<ProtoCCParser::Trans_bodyContext *> &ctx) {
    for (auto *transBodyCtx : ctx) {
      if (mlir::failed(mlirGen(transBodyCtx)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::Trans_bodyContext *ctx) {
    if (ctx->next_break()) {
      mlir::Location breakLoc = loc(*ctx->next_break()->BREAK());
      builder.create<mlir::pcc::BreakOp>(breakLoc);
      return mlir::success();
    }
    if (ctx->transaction())
      return mlirGen(ctx->transaction());

    if (ctx->network_send())
      return mlirGen(ctx->network_send());

    if (ctx->expressions())
      return mlirGen(ctx->expressions());

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::ConditionalContext *ctx) {
    if (ctx->if_stmt())
      return mlirGen(ctx->if_stmt());
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::If_stmtContext *ctx) {
    mlir::Location location = loc(*ctx->getStart());
    mlir::Value cond = mlirGen(ctx->cond_comb());
    bool hasElse = ctx->ELSE() != nullptr;
    auto ifOp = builder.create<mlir::pcc::IfOp>(location, cond, hasElse);

    builder.setInsertionPointToStart(ifOp.thenBlock());
    for (auto expr : ctx->if_expression()->exprwbreak()) {
      if (mlir::failed(mlirGen(expr)))
        return mlir::failure();
    }

    if (!ifOp.elseRegion().empty()) {
      builder.setInsertionPointToStart(ifOp.elseBlock());
    }

    builder.setInsertionPointAfter(ifOp);

    return mlir::success();
  }

  mlir::Value mlirGen(ProtoCCParser::Cond_combContext *ctx) {
    mlir::Location location = loc(*ctx->getStart());

    auto conditions = ctx->cond_rel();
    if (conditions.size() > 1) {
      auto combinatorials = ctx->combinatorial_operator();
      mlir::Value lhs_v = nullptr;
      mlir::Value rhs_v = nullptr;
      for (size_t i = 0; i < combinatorials.size(); i++) {
        if (!lhs_v) {
          auto lhs = conditions.at(i);
          lhs_v = mlirGen(lhs);
        }
        auto rhs = conditions.at(i + i); // This will always be defined
        rhs_v = mlirGen(rhs);

        if (combinatorials.at(i)->getText() == "&") {
          auto new_lhs =
              builder.create<mlir::pcc::LogicalAnd>(location, lhs_v, rhs_v);
          lhs_v = new_lhs; // move the lhs
        } else {
          auto new_lhs =
              builder.create<mlir::pcc::LogicalOr>(location, lhs_v, rhs_v);
          lhs_v = new_lhs; // move the lhs
        }
      }
      return lhs_v;
    } else {
      return mlirGen(conditions[0]);
    }
  }

  mlir::Value mlirGen(ProtoCCParser::Cond_relContext *ctx) {
    mlir::Location location = loc(*ctx->getStart());
    ProtoCCParser::Cond_selContext *selCtx = ctx->cond_sel();

    if (selCtx->cond_type_expr().size() == 1) {
      return mlirGen(selCtx->cond_type_expr(0));
    } else {
      mlir::Value lhs_v;
      mlir::Value rhs_v;
      for (size_t i = 0; i < selCtx->relational_operator().size(); i++) {
        if (!lhs_v) {
          lhs_v = mlirGen(selCtx->cond_type_expr(i));
        }
        rhs_v = mlirGen(selCtx->cond_type_expr(i + 1));

        std::string relOp = selCtx->relational_operator(i)->getText();
        if (relOp == "==") {
          auto rv = builder.create<mlir::pcc::EqualOp>(location, lhs_v, rhs_v);
          lhs_v = rv;
        }
      }
      return lhs_v;
    }
  }

  mlir::Value mlirGen(ProtoCCParser::Cond_type_exprContext *ctx) {
    mlir::Location location = loc(*ctx->getStart());
    if (ctx->cond_types().size() == 1) {
      return mlirGen(ctx->cond_types(0));
    } else {

      mlir::Value lhs_v;
      mlir::Value rhs_v;
      for (size_t i = 0; i < ctx->indv_math_op().size(); i++) {
        auto lhs = ctx->cond_types(i);
        auto rhs = ctx->cond_types(i + 1);
        if (!lhs_v) {
          lhs_v = mlirGen(lhs);
        }
        rhs_v = mlirGen(rhs);
        if (ctx->indv_math_op(i)->PLUS()) {
          mlir::Value rv = builder.create<mlir::pcc::AddOp>(
              location, rhs_v.getType(), lhs_v, rhs_v);
          lhs_v = rv;
        } else if (ctx->indv_math_op(i)->MINUS()) {
          mlir::Value rv = builder.create<mlir::pcc::SubOp>(
              location, rhs_v.getType(), lhs_v, rhs_v);
          lhs_v = rv;
        } else {
          mlir::Value rv = builder.create<mlir::pcc::MultOp>(
              location, rhs_v.getType(), lhs_v, rhs_v);
          lhs_v = rv;
        }
      }
      return lhs_v;
    }
  }

  mlir::Value mlirGen(ProtoCCParser::Cond_typesContext *ctx) {
    mlir::Location location = loc(*ctx->getStart());
    return builder.create<mlir::ConstantOp>(location,
                                            builder.getBoolAttr(true));
  }

  mlir::LogicalResult mlirGen(ProtoCCParser::ExprwbreakContext *ctx) {
    if (ctx->expressions())
      return mlirGen(ctx->expressions());
    if (ctx->network_send())
      return mlirGen(ctx->network_send());

    return mlir::success();
  }
};
} // namespace mlirGenImpl
namespace mlir{

namespace pcc {
mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx,
                       ProtoCCParser::DocumentContext *docCtx,
                       std::string filename) {
  return mlirGenImpl::MLIRGenImpl(mlirCtx).mlirGen(docCtx, std::move(filename));
}
} // namespace pcc
}
