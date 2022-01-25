#include "translation/murphi/codegen/MurphiStructs.h"
#include "translation/murphi/codegen/Boilerplate.h"
#include "translation/murphi/codegen/MurphiConstants.h"

namespace murphi {
namespace detail {

using namespace nlohmann;

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
          std::string paramType = "--fixme--";
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
        BinaryOps.minus}},
      {}};

  // ---- Line 3 if stmt
  // expr = i < cnt_fwd[n]-1
  BinaryExpr<ExprID, BinaryExpr<Designator, ExprID>> l3ifexpr = {
      {loopIV},
      {{CntKey + opf.netId, {Indexer{"array", ExprID{msg_p}}}},
       {"1"},
       BinaryOps.minus},
      BinaryOps.less_than};

  IfStmt<decltype(l3ifexpr)> l3ifstmt{l3ifexpr, {}, {}};

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

void to_json(json &j, const ReturnStmt &rs) {
  j = {{"typeId", "return"}, {"statement", {{"value", rs.value}}}};
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
      adr_a, {c_inmsg, {Indexer{"object", ExprID{c_adr}}}}, {}};

  // alias 2 -> cle: i_directory[m][adr]
  auto rhsalias = Designator{
      mach_prefix_v + mh.machId,
      {Indexer{"array", ExprID{c_mach}}, Indexer{"array", ExprID{c_adr}}}};
  auto cle_alias = detail::AliasStmt<decltype(rhsalias)>{cle_a, rhsalias, {}};

  /*
   * Switch Statement
   */
  // switch directory_entry.State
  json switch_stmt = boilerplate::getStateHandlerSwitch();
  switch_stmt["statement"]["cases"] = mh.cases;

  cle_alias.statements.emplace_back(switch_stmt);
  adr_alias.statements.emplace_back(cle_alias);

  auto returnTrueStmt = ReturnStmt{ExprID{"true"}};

  j = {{"procType", "function"},
       {"def",
        {{"id", detail::mach_handl_pref_f + mh.machId},
         {"params", {msg_param, mach_param}},
         {"returnType", detail::ID{detail::bool_t}},
         {"forwardDecls", {msg_fwd_decl}},
         {"statements", {adr_alias, std::move(returnTrueStmt)}}}}};
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
          BinaryOps.grtr_than},
      {},
      {}};

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

} // namespace detail
} // namespace murphi