#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <iostream>

namespace murphi {
/*
 * Implementation Details
 */
namespace detail {
/*
 * ConstDecl to_json helper method
 */
void to_json(json &j, const ConstDecl &c) {
  j = {{"id", c.id}, {"value", c.value}};
}
/*
 * Enum to_json helper method
 */
void to_json(json &j, const Enum &c) {
  j = {{"typeId", "enum"}, {"type", {{"decls", c.elems}}}};
}

/*
 * Union to_json helper method
 */
void to_json(json &j, const Union &u) {
  j = {{"typeId", "union"}, {"type", {{"listElems", u.elems}}}};
}

/*
 * Record to_json helper method
 */
void to_json(json &j, const Record &r) {
  json decls = json::array();
  std::for_each(std::begin(r.elems), std::end(r.elems), [&decls](auto &elem) {
    decls.push_back(
        {{"id", elem.first}, {"typeId", "ID"}, {"type", elem.second}});
  });
  j = {{"typeId", "record"}, {"type", {{"decls", decls}}}};
}

/*
 * ScalarSet to_json helper functions
 */
void to_json(json &j, const ScalarSet<std::string> &ss) {
  j = {{"typeId", "scalarset"}, {"type", {{"type", ss.value}}}};
}
void to_json(json &j, const ScalarSet<size_t> &ss) {
  size_t val = ss.value;
  j = {{"typeId", "scalarset"}, {"type", {{"type", val}}}};
}

void to_json(json &j, const ID &id) { j = {{"typeId", "ID"}, {"type", id.id}}; }

void to_json(json &j, const ExprID &id) {
  j = {{"typeId", "ID"}, {"expression", id.id}};
}

void to_json(json &j, const Designator &sd) {
  j = {{"typeId", "designator"},
       {"expression", {{"id", sd.id}, {"indexes", sd.indexes}}}};
}

void to_json(json &j, const Indexer &indexer) {
  j = {{"typeId", indexer.typeId}, {"index", indexer.index}};
}

void to_json(json &j, const NegExpr &negExpr) {
  j = {{"typeId", "neg_expr"}, {"expression", {{"expr", negExpr.expr}}}};
}

void to_json(json &j, const MessageFactory &m) {
  // bit of a hack here to remove the const from the parameter
  // it's a requirement of inja to have the parameter be const
  auto &messageFactory = const_cast<MessageFactory &>(m);
  // default params will be
  // adr
  Formal<ID> adr{detail::c_adr, {detail::ss_address_t}};
  // msgType
  Formal<ID> msgType{detail::c_mtype, {detail::e_message_type_t}};
  // src
  Formal<ID> src{detail::c_src, {detail::e_machines_t}};
  // dst
  Formal<ID> dst{detail::c_dst, {detail::e_machines_t}};

  std::vector<Formal<ID>> params = {std::move(adr), std::move(msgType),
                                    std::move(src), std::move(dst)};
  // add the default parameter assignments
  std::vector<Assignment<Designator, ExprID>> paramAssignments{
      {{c_msg, {Indexer{"object", ExprID{c_adr}}}}, {c_adr}},
      {{c_msg, {Indexer{"object", ExprID{c_mtype}}}}, {c_mtype}},
      {{c_msg, {Indexer{"object", ExprID{c_src}}}}, {c_src}},
      {{c_msg, {Indexer{"object", ExprID{c_dst}}}}, {c_dst}},
  };
  // additional params
  auto attrs = messageFactory.msgOp->getAttrs();
  std::for_each(
      std::begin(attrs), std::end(attrs), [&](mlir::NamedAttribute attr) {
        if (attr.first != c_adr && attr.first != c_mtype &&
            attr.first != c_src && attr.first != c_dst && attr.first != "id") {
          std::string paramName = attr.first.str();
          std::string paramType = MLIRTypeToMurphiTypeRef(
              attr.second.cast<mlir::TypeAttr>().getValue(), "");
          params.push_back({paramName, {paramType}});
          // TODO - possibly this could lead to incosistencies since we don't
          // un-define for the global msg type
          paramAssignments.push_back(
              {{c_msg, {Indexer{"object", ExprID{paramName}}}}, {paramName}});
        }
      });

  // TODO - make this return a ProcDecl instead of a ProcDef
  j = {{"id", messageFactory.msgOp.getMsgName().str()},
       {"params", params},
       {"forwardDecls",
        json::array({ForwardDecl<VarDecl<ID>>{"var", {c_msg, {r_message_t}}}})},
       {"returnType", detail::ID{detail::r_message_t}},
       {"statements", paramAssignments}};
}

void to_json(json &j, const OrderedSendFunction &usf) {
  constexpr char msg_p[] = "msg";
  // Line 1 -> Assert Statement (for too many messages in queue)
  // cnt_fwd[msg.dst]
  Designator net_count{
      CntKey + usf.netId,
      {Indexer{"array",
               Designator{msg_p, {Indexer{"object", ExprID{c_dst}}}}}}};

  BinaryExpr<decltype(net_count), ExprID> assert_val{
      net_count, {c_ordered_t}, BinaryOps.less_than};
  Assert<decltype(assert_val)> assert_stmt{assert_val, excess_messages_err};

  // Line 2 -> push message onto queue
  // fwd[msg.dst][cnt_fwd[msg.dst]] := msg;

  Designator lhs{
      usf.netId,
      {Indexer{"array", Designator{msg_p, {Indexer{"object", ExprID{c_dst}}}}},
       Indexer{"array",
               Designator{
                   CntKey + usf.netId,
                   {Indexer{"array", Designator{msg_p,
                                                {Indexer{"object",
                                                         ExprID{c_dst}}}}}}}}}};

  Assignment<decltype(lhs), ExprID> push_stmt{lhs, {msg_p}};

  // Line 3 -> increment count
  // cnt_fwd[msg.dst] := cnt_fwd[msg.dst] + 1;
  Designator cur_cnt{
      CntKey + usf.netId,
      {Indexer{"array",
               Designator{msg_p, {Indexer{"object", ExprID{c_dst}}}}}}};

  BinaryExpr<decltype(cur_cnt), ExprID> l3_rhs{cur_cnt, {"1"}, BinaryOps.plus};
  Assignment<decltype(cur_cnt), decltype(l3_rhs)> inc_count_stmt{cur_cnt,
                                                                 l3_rhs};

  j = {{"procType", "procedure"},
       {"def",
        {{"id", detail::send_pref_f + usf.netId},
         {"params",
          {detail::Formal<detail::ID>{msg_p, {{detail::r_message_t}}}}},
         {"statements", {assert_stmt, push_stmt, inc_count_stmt}}}}};
}

void to_json(json &j, const OrderedPopFunction &opf) {
  constexpr char msg_p[] = "n";

  // Line 1 -> assert stmt
  // Assert(cnt_fwd[n] > 0) "Trying to advance empty Q";
  Assert<BinaryExpr<Designator, ExprID>> line_1_assert{
      {{CntKey + opf.netId, {Indexer{"array", ExprID{msg_p}}}},
       {{"0"}},
       BinaryOps.grtr_than},
      ordered_pop_err};

  // line 2 for statement
  constexpr char loopIV[] = "i";
  ForStmt<ForRangeQuantifier<ExprID, BinaryExpr<Designator, ExprID>>> l2fstmt{
      {loopIV,
       {"0"},
       {{CntKey + opf.netId, {Indexer{"array", ExprID{msg_p}}}},
        {"1"},
        BinaryOps.minus}}};

  // ---- Line 3 if stmt
  // expr = i < cnt_fwd[n]-1
  BinaryExpr<ExprID, BinaryExpr<Designator, ExprID>> l3ifexpr = {
      {loopIV},
      {{CntKey + opf.netId, {Indexer{"array", ExprID{msg_p}}}},
       {"1"},
       BinaryOps.minus},
      BinaryOps.less_than};

  IfStmt<decltype(l3ifexpr)> l3ifstmt{l3ifexpr};

  // l4 -> move each element in buffer
  //  fwd[n][i] := fwd[n][i+1];
  Designator l4_lhs{
      opf.netId,
      {Indexer{"array", ExprID{msg_p}}, Indexer{"array", ExprID{loopIV}}}};

  Designator l4_rhs{opf.netId,
                    {Indexer{"array", ExprID{msg_p}},
                     Indexer{"array", BinaryExpr<ExprID, ExprID>{
                                          {loopIV}, {"1"}, BinaryOps.plus

                                      }}}};

  Assignment<decltype(l4_lhs), decltype(l4_rhs)> l4Ass{l4_lhs, l4_rhs};
  l3ifstmt.thenStmts.emplace_back(l4Ass);

  // l6 -> else un-define;
  UndefineStmt<decltype(l4_lhs)> l6undef{l4_lhs};
  l3ifstmt.elseStmts.emplace_back(l6undef);

  // nest if stmt within for loop
  l2fstmt.stmts.emplace_back(l3ifstmt);

  // l9 --> decrement count
  // cnt_fwd[n] := cnt_fwd[n] - 1;
  Assignment<Designator, BinaryExpr<Designator, ExprID>> l9dec{
      {CntKey + opf.netId, {Indexer{"array", ExprID{msg_p}}}},
      {{CntKey + opf.netId, {Indexer{"array", ExprID{msg_p}}}},
       {"1"},
       BinaryOps.minus}};

  j = {{"procType", "procedure"},
       {"def",
        {

            {"id", detail::pop_pref_f + opf.netId},
            {"params",
             {detail::Formal<detail::ID>{msg_p, {{detail::e_machines_t}}}}},
            {"statements", {line_1_assert, l2fstmt, l9dec}}}}};
}

void to_json(json &j, const UnorderedSendFunction &usf) {
  constexpr char msg_p[] = "msg";
  // L1 -> assert check not full
  // Assert (MultiSetCount(i:resp[msg.dst], true) < U_NET_MAX) "Too many
  // messages";
  MultisetCount ms_count{
      "i",
      Designator{
          usf.netId,
          {Indexer{"array",
                   Designator{msg_p, {Indexer{"object", ExprID{c_dst}}}}}}},
      detail::ExprID{"true"}};
  BinaryExpr<decltype(ms_count), ExprID> assert_val{
      ms_count, {"true"}, BinaryOps.less_than};
  Assert<decltype(assert_val)> l1Ass{assert_val, excess_messages_err};

  // L2 -> MultiSetAdd(msg, req[msg.dst]);
  ExprID firsParam{msg_p};
  Designator secParam{
      usf.netId,
      {Indexer{"array",
               Designator{msg_p, {Indexer{"object", ExprID{c_dst}}}}}}};
  ProcCall l2add{multiset_add_f, {firsParam, secParam}};

  j = {{"procType", "procedure"},
       {"def",
        {{"id", detail::send_pref_f + usf.netId},
         {"params", {detail::Formal<detail::ID>{msg_p, {detail::r_message_t}}}},
         {"statements", {l1Ass, l2add}}}}};
}

void to_json(json &j, const MultisetCount &ms) {
  j = {{"typeId", "ms_count"},
       {"expression",
        {{"varId", ms.varId},
         {"varValue", ms.varValue},
         {"predicate", ms.predicate}}}};
}

void to_json(json &j, const ProcCall &fn) {
  j = {{"typeId", "proc_call"},
       {"statement", {{"funId", fn.funId}, {"actuals", fn.actuals}}}};
}

void to_json(json &j, const ProcCallExpr &fn) {
  j = {{"typeId", "proc_call"},
       {"expression", {{"funId", fn.funId}, {"actuals", fn.actuals}}}};
}

void to_json(json &j, const CaseStmt &caseStmt) {
  j = {{"expr", caseStmt.expr}, {"statements", caseStmt.statements}};
}

void to_json(json &j, const SwitchStmt &sw) {
  j = {{"typeId", "switch"},
       {"statement",
        {{"expr", sw.expr},
         {"cases", sw.cases},
         {"elseStatements", sw.elseStatements}}}};
}
///*** Machine Handler ***///
void to_json(json &j, const detail::MachineHandler &mh) {
  /*
   * Parameters
   */
  // Param 1 -> inmsg:Message
  auto msg_param = detail::Formal<ID>{c_inmsg, {{r_message_t}}};
  // Param 2 -> m:OBJSET_directory
  auto mach_param = detail::Formal<ID>{c_mach, {{SetKey + mh.machId}}};

  /*
   * Forward Decls
   */
  // var msg:Message;
  constexpr char msg_var[] = "msg";
  auto msg_fwd_decl = detail::ForwardDecl<detail::VarDecl<detail::ID>>{
      "var", {msg_var, {r_message_t}}};

  /*
   * Alias Stmts
   */
  // alias 1 -> alias adr: inmsg.adr do
  auto adr_alias = detail::AliasStmt<Designator>{
      adr_a, {c_inmsg, {Indexer{"object", ExprID{c_adr}}}}};

  // alias 2 -> cle: i_directory[m][adr]
  auto rhsalias = Designator{
      mach_prefix_v + mh.machId,
      {Indexer{"array", ExprID{c_mach}}, Indexer{"array", ExprID{c_adr}}}};
  auto cle_alias = detail::AliasStmt<decltype(rhsalias)>{cle_a, rhsalias};

  /*
   * Switch Statement
   */
  // switch directory_entry.State
  auto switch_stmt = detail::SwitchStmt{
      Designator{cle_a, {Indexer{"object", ExprID{c_state}}}}};

  cle_alias.statements.emplace_back(switch_stmt);
  adr_alias.statements.emplace_back(cle_alias);

  j = {{"procType", "function"},
       {"def",
        {{"id", detail::mach_handl_pref_f + mh.machId},
         {"params", {msg_param, mach_param}},
         {"returnType", detail::ID{detail::bool_t}},
         {"forwardDecls", {msg_fwd_decl}},
         {"statements", {adr_alias}}}}};
}

/*** CPU Event Handler ***/

void to_json(json &j, const CPUEventHandler &cpuEventHandler) {

  /*
   * Params
   */
  auto adr_param = Formal<ID>{c_adr, {ss_address_t}};
  auto cache_param = Formal<ID>{c_mach, {SetKey + machines.cache.str()}};

  /*
   * Forward Decls
   */
  auto msg_fwd_decl = detail::ForwardDecl<detail::VarDecl<detail::ID>>{
      "var", {c_msg, {r_message_t}}};

  /*
   * Alias Stmt
   */
  // alias cle: i_cache[m][adr]
  auto alias_expr = Designator{
      mach_prefix_v + machines.cache.str(),
      {Indexer{"array", ExprID{c_mach}}, Indexer{"array", ExprID{c_adr}}}};

  auto alias_stmt = AliasStmt<decltype(alias_expr)>{cle_a, alias_expr,
                                                    cpuEventHandler.statements};

  j = {{"procType", "procedure"},
       {"def",
        {{"id", cpu_action_pref_f + cpuEventHandler.start_state},
         {"params", {adr_param, cache_param}},
         {"forwardDecls", {msg_fwd_decl}},
         {"statements", {alias_stmt}}}}};
}

///*** Cache Event Handler ***///
void to_json(json &j, const CacheRuleHandler &ceh) {
  /*
   * Quantifiers
   */

  // Q1 -> m:OBJSET_cache
  auto q1 = ForEachQuantifier<ID>{c_mach, {SetKey + machines.cache.str()}};

  // Q2 -> adr:Address
  auto q2 = ForEachQuantifier<ID>{c_adr, {ss_address_t}};

  /*
   * Alias(s)
   */
  // A1 --> alias cle:i_cache[m][adr]
  auto alias_expr = Designator{
      mach_prefix_v + machines.cache.str(),
      {Indexer{"array", ExprID{c_mach}}, Indexer{"array", ExprID{c_adr}}}};
  auto alias_rule = AliasRule{cle_a, alias_expr, ceh.rules};

  j = {{"typeId", "ruleset"},
       {"rule", {{"quantifiers", {q1, q2}}, {"rules", {alias_rule}}}}};
}

// *** CPUEventRule *** ///
void to_json(json &j, const CPUEventRule &er) {
  auto ruleDesc = er.state + "_" + er.event;
  auto ruleExpr = BinaryExpr<Designator, ExprID>{
      {cle_a, {Indexer{"object", ExprID{c_state}}}}, {er.state}, BinaryOps.eq};
  auto ruleStatement =
      ProcCall{cpu_action_pref_f + ruleDesc, {ExprID{adr_a}, ExprID{c_mach}}};

  j = SimpleRule{ruleDesc, ruleExpr, {}, {ruleStatement}};
}

// *** OrderedRuleset *** //
void to_json(json &j, const OrderedRuleset &orderedRuleset) {
  constexpr auto mach_q = "n";
  /*
   * Quantifier
   */
  // n:Machines
  auto ruleset_quant = detail::ForEachQuantifier<ID>{mach_q, {e_machines_t}};

  /*
   * Alias
   */
  // msg:fwd[n][0]
  auto alias = AliasRule{c_msg,
                         Designator{orderedRuleset.netId,
                                    {Indexer{"array", ExprID{mach_q}},
                                     Indexer{"array", ExprID{"0"}}}},
                         {}};

  /*
   * Network Rule
   */
  // cnt_fwd[n] > 0
  constexpr auto net_rule_name_pref = "Receive ";
  auto net_rule = SimpleRule{
      net_rule_name_pref + orderedRuleset.netId,
      BinaryExpr<Designator, ExprID>{
          {CntKey + orderedRuleset.netId, {Indexer{"array", ExprID{mach_q}}}},
          {"0"},
          BinaryOps.grtr_than}};

  auto get_inner = [&](const std::string &mach) -> json {
    return IfStmt<ProcCallExpr>{
        {mach_handl_pref_f + mach, {ExprID{c_msg}, ExprID{mach_q}}},
        {ProcCall{pop_pref_f + orderedRuleset.netId, {ExprID{mach_q}}}},
        {}};
  };

  /*
   * Call Correct Machine Handler
   */
  auto if_member = IfStmt<ProcCallExpr>{
      {is_member_f, {ExprID{mach_q}, ExprID{directory_set_t()}}},
      {get_inner(machines.directory.str())},
      {get_inner(machines.cache.str())}};

  net_rule.statements.emplace_back(if_member);
  alias.rules.emplace_back(net_rule);
  j = detail::RuleSet{{ruleset_quant}, {alias}};
}

void to_json(json &j, const UnorderedRuleset &urs) {
  constexpr auto ms_idx = "midx";
  constexpr auto ruleset_idx = "n";
  constexpr auto mach_alias = "mach";
  constexpr auto msg_alias = "msg";

  auto get_inner = [](const std::string &mach) -> json {
    return IfStmt<ProcCallExpr>{
        {mach_handl_pref_f + mach, {ExprID{msg_alias}, ExprID{ruleset_idx}}},
        {ProcCall{multiset_remove_f, {ExprID{ms_idx}, ExprID{mach_alias}}}},
        {}};
  };

  auto network_rule = SimpleRule{
      "Receive " + urs.netId,
      NegExpr{ProcCallExpr{
          is_undefined_f,
          {Designator{msg_alias, {Indexer{"object", ExprID{c_mtype}}}}}}},
      {},
      {IfStmt<ProcCallExpr>{
          {is_member_f, {ExprID{ruleset_idx}, ExprID{directory_set_t()}}},
          {get_inner(machines.directory.str())},
          {get_inner(machines.cache.str())}}}};

  auto msg_aliasrule =
      AliasRule{msg_alias,
                Designator{mach_alias, {Indexer{"array", ExprID{ms_idx}}}},
                {network_rule}};

  auto mach_aliasrule =
      AliasRule{mach_alias,
                Designator{urs.netId, {Indexer{"array", ExprID{ruleset_idx}}}},
                {msg_aliasrule}};

  auto choose_rule =
      ChooseRule{ms_idx,
                 Designator{urs.netId, {Indexer{"array", ExprID{ruleset_idx}}}},
                 {mach_aliasrule}};

  auto ruleset = RuleSet{{ForEachQuantifier<ID>{ruleset_idx, {e_machines_t}}},
                         {choose_rule}};

  j = ruleset;
}
/*
 * Rules
 */

/// *** SIMPLE RULE ***///

void to_json(json &j, const SimpleRule &sr) {
  j = {{"typeId", "simple_rule"},
       {"rule",
        {{"ruleDesc", sr.ruleDesc},
         {"expr", sr.expr},
         {"decls", sr.decls},
         {"statements", sr.statements}}}};
}

///*** RuleSet ***///
void to_json(json &j, const RuleSet &rs) {
  j = {{"typeId", "ruleset"},
       {"rule", {{"quantifiers", rs.quantifiers}, {"rules", rs.rules}}}};
}

///*** AliasRule ***///
void to_json(json &j, const AliasRule &ar) {
  j = {{"typeId", "alias_rule"},
       {"rule", {{"id", ar.id}, {"expr", ar.expr}, {"rules", ar.rules}}}};
}

///*** ChooseRule ***///
void to_json(json &j, const ChooseRule &cr) {
  j = {{"typeId", "choose_rule"},
       {"rule", {{"index", cr.index}, {"expr", cr.expr}, {"rules", cr.rules}}}};
}

///*** StartState ***///
void to_json(json &j, const StartState &ss) {
  j = {{"typeId", "start_state"},
       {"rule",
        {{"desc", ss.desc},
         {"decls", ss.decls},
         {"statements", ss.statements}}}};
}

/*
 * Network Decl helper
 */
json emitNetworkDefinitionJson() {
  auto ord_type_name = std::string(ObjKey) + ordered;
  auto ord_type_count_name = std::string(ObjKey) + ordered_cnt;
  auto un_ord_type_name = std::string(ObjKey) + unordered;
  detail::TypeDecl<detail::Array<
      detail::ID,
      detail::Array<detail::SubRange<size_t, std::string>, detail::ID>>>
      ordered_network_type{
          ord_type_name,
          {{e_machines_t},
           {{0, std::string(c_ordered_t) + "-1"}, {r_message_t}}}};
  detail::TypeDecl<
      detail::Array<detail::ID, detail::SubRange<size_t, std::string>>>
      ordered_network_count{ord_type_name, {{e_machines_t}, {0, c_ordered_t}}};
  detail::TypeDecl<
      detail::Array<detail::ID, detail::Multiset<detail::ID, detail::ID>>>
      unordered_network_t{un_ord_type_name,
                          {{e_machines_t}, {{c_unordered_t}, {r_message_t}}}};
  json j = {ordered_network_type, ordered_network_count, unordered_network_t};
  return j;
}

bool validateMurphiJSON(const json &j) {
  const std::string schema_path =
      std::string(JSONValidation::schema_base_directory) +
      "gen_Murphi_json_schema.json";
  return JSONValidation::validate_json(schema_path, j);
}
std::string e_directory_state_t() {
  return machines.directory.str() + state_suffix;
}
std::string e_cache_state_t() {
  return machines.cache.str() + state_suffix;
}
std::string r_cache_entry_t() {
  return std::string(EntryKey) + machines.cache.str();
}
std::string r_directory_entry_t() {
  return std::string(EntryKey) + machines.directory.str();
}

std::string cache_v() { return mach_prefix_v + machines.cache.str(); }
std::string directory_v() {
  return mach_prefix_v + machines.directory.str();
}

} // namespace detail

mlir::LogicalResult MurphiCodeGen::translate() {
  generateConstants();
  generateTypes();
  generateVars();
  generateMethods();
  generateRules();
  return render();
}

bool MurphiCodeGen::is_json_valid() { return detail::validateMurphiJSON(data); }

void MurphiCodeGen::generateConstants() {
  // add the defined constants
  for (auto constantOp : moduleInterpreter.getConstants()) {
    data["decls"]["const_decls"].push_back(
        detail::ConstDecl{constantOp.id().str(), constantOp.val()});
  }

  // boilerplate constants
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_val_cnt_t, detail::c_val_max});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_adr_cnt_t, detail::c_adr_cnt});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_ordered_t, detail::c_ordered_size});
  data["decls"]["const_-decls"].push_back(
      detail::ConstDecl{detail::c_unordered_t, detail::c_unordered_size});
}

void MurphiCodeGen::generateTypes() {
  _typeEnums();
  _typeStatics();
  _typeMachineSets();
  _typeMessage();
  _typeMachines();
  _typeNetworkObjects();
  _typeMutexes();
}

mlir::LogicalResult MurphiCodeGen::render() {
  // validate json
  assert(detail::validateMurphiJSON(data) &&
         "JSON from codegen does not validate with the json schema");
  auto &env = InjaEnvSingleton::getInstance();
  auto tmpl = env.parse_template("murphi_base.tmpl");
  auto result = env.render(tmpl, data);
  output << result;
  return mlir::success();
}

void MurphiCodeGen::_typeEnums() {
  // access enum
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
      detail::e_access_t, detail::Enum{{"none", "load", "store"}}});
  // messages enum
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
      detail::e_message_type_t, {moduleInterpreter.getEnumMessageTypes()}});
  // cache state
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
      detail::e_cache_state_t(),
      {moduleInterpreter.getEnumMachineStates(detail::machines.cache.str())}});
  data["decls"]["type_decls"].push_back(
      detail::TypeDecl<detail::Enum>{detail::e_directory_state_t(),
                                     {moduleInterpreter.getEnumMachineStates(
                                         detail::machines.directory.str())}});
}

void MurphiCodeGen::_typeStatics() {
  // Address type
  data["decls"]["type_decls"].push_back(
      detail::TypeDecl<detail::ScalarSet<std::string>>{
          detail::ss_address_t,
          detail::ScalarSet<std::string>{detail::c_val_cnt_t}});
  // ClValue type
  data["decls"]["type_decls"].push_back(
      detail::TypeDecl<detail::SubRange<size_t, std::string>>{
          detail::sr_cache_val_t, {0, detail::c_val_cnt_t}});
}

void MurphiCodeGen::_typeMachineSets() {
  auto add_mach_decl_type = [&](const std::string &machine,
                                const mlir::pcc::PCCType &type) {
    std::string typeIdent = machine == detail::machines.cache
                                ? detail::cache_set_t()
                                : detail::directory_set_t();
    if (type.isa<mlir::pcc::SetType>()) {
      auto setSize = type.cast<mlir::pcc::SetType>().getNumElements();
      data["decls"]["type_decls"].push_back(
          detail::TypeDecl<detail::ScalarSet<size_t>>{
              detail::cache_set_t(), detail::ScalarSet<size_t>{setSize}});
    } else {
      // must be a struct type
      assert(type.isa<mlir::pcc::StructType>() &&
             "machine was not of set/struct type!");
      data["decls"]["type_decls"].push_back(
          detail::TypeDecl<detail::Enum>{typeIdent, {{machine}}});
    }
  };

  // add the cache set
  add_mach_decl_type(detail::machines.cache.str(),
                     moduleInterpreter.getCache().getType());
  add_mach_decl_type(detail::machines.directory.str(),
                     moduleInterpreter.getDirectory().getType());

  // push a union of these for the Machines type
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Union>{
      detail::e_machines_t,
      {{detail::cache_set_t(), detail::directory_set_t()}}});
}

void MurphiCodeGen::_typeNetworkObjects() {
  for (const auto &type : detail::emitNetworkDefinitionJson()) {
    data["decls"]["type_decls"].push_back(type);
  }
}

void MurphiCodeGen::_typeMachines() {
  // *** cache *** //
  mlir::pcc::CacheDeclOp cacheOp = moduleInterpreter.getCache();
  _getMachineEntry(cacheOp.getOperation());

  // *** directory *** //
  mlir::pcc::DirectoryDeclOp directoryOp = moduleInterpreter.getDirectory();

  _getMachineEntry(directoryOp.getOperation());
  _getMachineMach();
  _getMachineObjs();
}

void MurphiCodeGen::_getMachineEntry(mlir::Operation *machineOp) {
  // Check that the operation is either cache or directory decl
  const auto opIdent = machineOp->getName().getIdentifier().strref();
  assert((opIdent == detail::opStringMap.cache_decl ||
          opIdent == detail::opStringMap.dir_decl) &&
         "invalid operation passed to generation machine function");
  const auto machineIdent = opIdent == detail::opStringMap.cache_decl
                                ? detail::machines.cache
                                : detail::machines.directory;

  const auto is_id_attr = [](const mlir::NamedAttribute &attr) {
    return attr.first == "id";
  };

  std::vector<std::pair<std::string, std::string>> record_elems;

  const auto generate_mach_attr_field =
      [&record_elems, &machineIdent](const mlir::NamedAttribute &attr) {
        std::string fieldID = attr.first.str();
        mlir::TypeAttr typeAttr = attr.second.cast<mlir::TypeAttr>();
        std::string fieldType =
            MLIRTypeToMurphiTypeRef(typeAttr.getValue(), machineIdent);
        record_elems.emplace_back(fieldID, fieldType);
      };

  // for each attribute; generate a machine attribute
  std::for_each(machineOp->getAttrs().begin(), machineOp->getAttrs().end(),
                [&](const mlir::NamedAttribute &attribute) {
                  const mlir::NamedAttribute named_attr =
                      static_cast<mlir::NamedAttribute>(attribute);
                  if (!is_id_attr(named_attr)) {
                    generate_mach_attr_field(named_attr);
                  }
                });

  std::string machine_id_t = machineIdent == detail::machines.cache
                                 ? detail::r_cache_entry_t()
                                 : detail::r_directory_entry_t();
  // generate the correct murphi declaration
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Record>{
      std::move(machine_id_t), {std::move(record_elems)}});
}

void MurphiCodeGen::_getMachineMach() {

  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> cacheMach{
      detail::cache_mach_t(),
      {{detail::ss_address_t}, {detail::cache_set_t()}}};

  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> dirMach{
      detail::directory_mach_t(),
      {{detail::ss_address_t}, {detail::directory_set_t()}}};

  data["decls"]["type_decls"].push_back(std::move(cacheMach));
  data["decls"]["type_decls"].push_back(std::move(dirMach));
}

void MurphiCodeGen::_getMachineObjs() {

  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> cacheObj{
      detail::cache_obj_t(),
      {{detail::cache_set_t()}, {detail::cache_mach_t()}}};

  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> dirObj{
      detail::directory_obj_t(),
      {{detail::directory_set_t()}, {detail::directory_mach_t()}}};

  data["decls"]["type_decls"].push_back(std::move(cacheObj));
  data["decls"]["type_decls"].push_back(std::move(dirObj));
}

void MurphiCodeGen::_typeMessage() {
  // generate the glob msg
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Record>{
      detail::r_message_t, {get_glob_msg_type()}});
}

void MurphiCodeGen::_typeMutexes() {
  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> mutex_t{
      detail::a_cl_mutex_t, {{detail::ss_address_t}, {"boolean"}}};
  data["decls"]["type_decls"].push_back(std::move(mutex_t));
}

std::vector<std::pair<std::string, std::string>>
MurphiCodeGen::get_glob_msg_type() {
  std::map<std::string, std::string> glob_msg_type;
  auto all_msg_types = moduleInterpreter.getMessages();

  std::for_each(
      all_msg_types.begin(), all_msg_types.end(),
      [&](const mlir::pcc::MsgDeclOp &msgDeclOp) {
        std::for_each(
            msgDeclOp->getAttrs().begin(), msgDeclOp->getAttrs().end(),
            [&](const mlir::NamedAttribute &named_attr) {
              if (named_attr.first != "id") { // skip the id field
                mlir::TypeAttr typeAttr =
                    named_attr.second.cast<mlir::TypeAttr>();
                glob_msg_type.insert(
                    {named_attr.first.str(),
                     MLIRTypeToMurphiTypeRef(typeAttr.getValue(), "")});
              }
            });
      });

  // copy to vector
  std::vector<std::pair<std::string, std::string>> msg_decl_type;
  msg_decl_type.reserve(glob_msg_type.size() + 1);
  for (auto &mv : glob_msg_type) {
    msg_decl_type.emplace_back(mv);
  }

  // push back the address type also
  msg_decl_type.emplace_back(detail::c_adr, detail::ss_address_t);

  return msg_decl_type;
}

// Free Functions
std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t,
                                    const llvm::StringRef mach) {
  if (t.isa<mlir::pcc::DataType>()) {
    return detail::sr_cache_val_t;
  }
  if (t.isa<mlir::pcc::StateType>()) {
    return mach == detail::machines.cache ? detail::e_cache_state_t()
                                          : detail::e_directory_state_t();
  }
  if (t.isa<mlir::pcc::IDType>()) {
    return detail::e_machines_t;
  }
  if (t.isa<mlir::pcc::MsgIdType>()) {
    return detail::r_message_t;
  }
  // TODO - fix this implementation
  // we're better off if we declare a type for this first
  // and then refer to it.
  if (t.isa<mlir::pcc::SetType>()) {
    // create a type that we can refer to. This may be more challenging now
    mlir::pcc::SetType st = t.cast<mlir::pcc::SetType>();

    return "multiset [ " + std::to_string(st.getNumElements()) + " ] of " +
           MLIRTypeToMurphiTypeRef(st.getElementType(), "");
  }

  if (t.isa<mlir::pcc::IntRangeType>()) {
    mlir::pcc::IntRangeType intRangeType = t.cast<mlir::pcc::IntRangeType>();
    return std::to_string(intRangeType.getStartRange()) + ".." +
           std::to_string(intRangeType.getEndRange());
  }

  // TODO - add support for more types
  assert(0 && "currently using an unsupported type!");
}

void MurphiCodeGen::generateVars() {
  _varMachines();
  _varNetworks();
  _varMutexes();
}

void MurphiCodeGen::_varMachines() {

  detail::VarDecl<detail::ID> var_cache{detail::cache_v(),
                                        {{detail::cache_obj_t()}}};
  detail::VarDecl<detail::ID> var_dir{detail::directory_v(),
                                      {{detail::directory_obj_t()}}};
  data["decls"]["var_decls"].push_back(var_cache);
  data["decls"]["var_decls"].push_back(var_dir);
}
void MurphiCodeGen::_varNetworks() {
  auto ops = moduleInterpreter.getOperations<mlir::pcc::NetDeclOp>();
  std::for_each(ops.begin(), ops.end(), [&](mlir::pcc::NetDeclOp &netDeclOp) {
    if (netDeclOp.getType().getOrdering() == "ordered") {

      detail::VarDecl<detail::ID> ord_net_v{
          netDeclOp.netId().str(),
          {{std::string(detail::ObjKey) + detail::ordered}}};
      detail::VarDecl<detail::ID> ord_net_cnt_v{
          std::string(detail::CntKey) + netDeclOp.netId().str(),
          {{std::string(detail::ObjKey) + detail::ordered_cnt}}};
      data["decls"]["var_decls"].push_back(ord_net_v);
      data["decls"]["var_decls"].push_back(ord_net_cnt_v);

    } else {
      detail::VarDecl<detail::ID> unord_net_v{
          netDeclOp.netId().str(),
          {{std::string(detail::ObjKey) + detail::unordered}}};
      data["decls"]["var_decls"].push_back(unord_net_v);
    }
  });
}
void MurphiCodeGen::_varMutexes() {
  detail::VarDecl<detail::ID> mutex_v{detail::cl_mutex_v,
                                      {{detail::a_cl_mutex_t}}};
  data["decls"]["var_decls"].push_back(mutex_v);
}

/*
 * Murphi Methods
 */

void MurphiCodeGen::generateMethods() {
  _generateMsgFactories();
  _generateHelperFunctions();
}

/*
 * MESSAGE FACTORIES
 */

void MurphiCodeGen::_generateMsgFactories() {
  auto msgDecls = moduleInterpreter.getOperations<mlir::pcc::MsgDeclOp>();
  std::for_each(std::begin(msgDecls), std::end(msgDecls),
                [&](mlir::pcc::MsgDeclOp msgDeclOp) {
                  data["proc_decls"].push_back(
                      {{"procType", "function"},
                       {"def", detail::MessageFactory{msgDeclOp}}});
                });
}

/*
 * Helper Functions
 */

void MurphiCodeGen::_generateHelperFunctions() {
  _generateMutexHelpers();
  _generateSendPopFunctions();
  _generateMachineHandlers();
  _generateCPUEventHandlers();
}

void MurphiCodeGen::_generateMutexHelpers() {
  constexpr char adr_param[] = "adr";

  // ACQUIRE MUTEX
  json acq_mut_proc = {
      {"id", detail::aq_mut_f},
      {"params", json::array({detail::Formal<detail::ID>{
                     adr_param, {{detail::ss_address_t}}}})},
      {"statements",
       json::array({detail::Assignment<detail::Designator, detail::ExprID>{
           {detail::cl_mutex_v,
            {detail::Indexer{"array", detail::ExprID{adr_param}}}},
           {"true"}}

       })}};
  json acq_proc_decl = {{"procType", "procedure"},
                        {"def", std::move(acq_mut_proc)}};

  // RELEASE MUTEX
  json rel_mut_proc = {
      {"id", detail::rel_mut_f},
      {"params", json::array({detail::Formal<detail::ID>{
                     adr_param, {{detail::ss_address_t}}}})},
      {"statements",
       json::array({detail::Assignment<detail::Designator, detail::ExprID>{
           {detail::cl_mutex_v,
            {detail::Indexer{"array", detail::ExprID{adr_param}}}},
           {"false"}}

       })}};
  json rel_proc_decl = {{"procType", "procedure"},
                        {"def", std::move(rel_mut_proc)}};

  // Add them to the json structure
  data["proc_decls"].push_back(std::move(acq_proc_decl));
  data["proc_decls"].push_back(std::move(rel_proc_decl));
}

void MurphiCodeGen::_generateSendPopFunctions() {
  auto networks = moduleInterpreter.getNetworks();
  std::for_each(
      networks.begin(), networks.end(), [&](mlir::pcc::NetDeclOp netDeclOp) {
        auto netID = netDeclOp.netId();
        auto order = netDeclOp.result()
                         .getType()
                         .cast<mlir::pcc::NetworkType>()
                         .getOrdering();
        if (order == "ordered") {
          // ordered networks
          data["proc_decls"].push_back(
              detail::OrderedSendFunction{netID.str()});
          data["proc_decls"].push_back(detail::OrderedPopFunction{netID.str()});
        } else {
          data["proc_decls"].push_back(
              detail::UnorderedSendFunction{netID.str()});
        }
      });
}

/*
 * Machine Handlers
 */

void MurphiCodeGen::_generateMachineHandlers() {
  // cache
  // TODO - generate the inner states

  auto cacheHandler = detail::MachineHandler{detail::machines.cache.str(), {}};
  data["proc_decls"].push_back(cacheHandler);

  // directory
  // TODO - generate directory inner states
  auto directoryHandler =
      detail::MachineHandler{detail::machines.directory.str(), {}};
  data["proc_decls"].push_back(directoryHandler);
}

/*
 * CPU Event Handler
 */

void MurphiCodeGen::_generateCPUEventHandlers() {
  // For Each stable state generate CPU Event handlers for load/store/evict
  // I -> Special state -> NO EVICT
  auto stableStates = std::vector<std::string>{"cache_I", "cache_S", "cache_M"};

  for (auto &ss : stableStates) {
    for (auto &cpu_event : detail::cpu_events) {
      if (cpu_event != "evict" || ss != "cache_I") {
        auto cpu_event_handler =
            detail::CPUEventHandler{ss + "_" + cpu_event.str(), {}};
        data["proc_decls"].push_back(cpu_event_handler);
      }
    }
  }
}

/*
 * Rules
 */

void MurphiCodeGen::generateRules() {
  _generateCacheRuleHandler();
  _generateNetworkRules();
  _generateStartState();
}

void MurphiCodeGen::_generateCacheRuleHandler() {
  detail::CacheRuleHandler cacheRuleHandler;
  const auto stableStates =
      std::vector<std::string>{"cache_I", "cache_S", "cache_M"};

  for (auto &ss : stableStates) {
    for (auto cpu_event : detail::cpu_events) {
      if (cpu_event != "evict" || ss != "cache_I") {
        auto cpuEventRule = detail::CPUEventRule{ss, cpu_event.str()};
        cacheRuleHandler.rules.emplace_back(std::move(cpuEventRule));
      }
    }
  }
  data["rules"].push_back(std::move(cacheRuleHandler));
}

void MurphiCodeGen::_generateNetworkRules() {
  auto networks = moduleInterpreter.getNetworks();
  for (auto &nw : networks) {
    if (nw.getType().getOrdering() == "ordered") {
      data["rules"].push_back(detail::OrderedRuleset{nw.netId().str()});
    } else {
      data["rules"].push_back(detail::UnorderedRuleset{nw.netId().str()});
    }
  }
}

void MurphiCodeGen::_generateStartState() {
  auto ss = detail::StartState{"murphi start state", {}, {}};
  ///*** Initialize the Directory and Cache objects ***///
  auto mach_ss_stmts = [&](const std::string &machId) -> json {
    constexpr auto mach_idx = "i";
    constexpr auto adr_idx = "a";
    auto for_mach = detail::ForStmt<detail::ForEachQuantifier<detail::ID>>{
        {mach_idx, {detail::SetKey + machId}}, {}};

    auto for_adr = detail::ForStmt<detail::ForEachQuantifier<detail::ID>>{
        {adr_idx, {detail::ss_address_t}}, {}};

    // for each value in the struct setup a default;
    // i_cache[i][a].??? := cache_I;
    // TODO - implement this properly
    auto decl = detail::ExprID{"State"};
    auto default_value = detail::ExprID{machId + "_I"};

    auto common_start =
        detail::Designator{detail::mach_prefix_v + machId,
                           {detail::Indexer{"array", detail::ExprID{mach_idx}},
                            detail::Indexer{"array", detail::ExprID{adr_idx}}}};

    auto lhs = common_start; // copy the start
    lhs.indexes.emplace_back(detail::Indexer{"object", detail::ExprID{decl}});

    for_adr.stmts.emplace_back(
        detail::Assignment<decltype(lhs), decltype(default_value)>{
            lhs, default_value});
    for_mach.stmts.emplace_back(for_adr);
    return for_mach;
  };
  ss.statements.push_back(mach_ss_stmts(detail::machines.cache.str()));
  ss.statements.push_back(mach_ss_stmts(detail::machines.directory.str()));

  /// *** Initialize Mutexes *** ///
  auto generate_mutex_inits = []() -> json {
    constexpr auto adr_idx = "a";
    auto mut_false = detail::Assignment<detail::Designator, detail::ExprID>{
        {detail::cl_mutex_v,
         {detail::Indexer{"array", detail::ExprID{adr_idx}}}},
        {"false"}};
    auto for_adr = detail::ForStmt<detail::ForEachQuantifier<detail::ID>>{
        {adr_idx, {detail::ss_address_t}}, {std::move(mut_false)}};
    return for_adr;
  };
  ss.statements.emplace_back(generate_mutex_inits());

  /// ** Undefine networks //
  for (auto nw : moduleInterpreter.getNetworks()) {
    auto netId = nw.netId().str();
    auto rhs = detail::Designator{netId, {}};
    ss.statements.emplace_back(detail::UndefineStmt<decltype(rhs)>{rhs});
  }

  /// ***  Set all ordered counts to zero  *** ///
  auto gen_ordered_cnt_ss = [](llvm::StringRef netId) -> json {
    constexpr auto mach_idx = "n";
    auto for_quant =
        detail::ForEachQuantifier<detail::ID>{mach_idx, {detail::e_machines_t}};

    auto cnt_stmt = detail::Assignment<detail::Designator, detail::ExprID>{
        {detail::CntKey + netId.str(),
         {detail::Indexer{"array", detail::ExprID{mach_idx}}}},
        {"0"}};

    return detail::ForStmt<decltype(for_quant)>{std::move(for_quant),
                                                {std::move(cnt_stmt)}};
  };

  for (auto &nw : moduleInterpreter.getNetworks()) {
    if (nw.getType().getOrdering() == "ordered") {
      ss.statements.emplace_back(gen_ordered_cnt_ss(nw.netId()));
    }
  }

  data["rules"].push_back(ss);
}

} // namespace murphi
