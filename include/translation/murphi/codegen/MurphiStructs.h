#pragma once
#include "nlohmann/json.hpp"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>
// TODO - remove this header
#include "PCC/PCCOps.h"

namespace murphi {
namespace detail {

/*
 * Helper Structs to generate JSON
 */

// *** ConstDecl
struct ConstDecl {
  std::string id;
  size_t value;
};

void to_json(nlohmann::json &j, const ConstDecl &c);

// *** TypeDecl
template <class TypeT> struct TypeDecl {
  std::string id;
  TypeT type;
};

template <class TypeT>
void to_json(nlohmann::json &j, const TypeDecl<TypeT> &typeDecl) {
  j = typeDecl.type;
  j["id"] = typeDecl.id;
}

// var_decl uses the same underlying struct
template <class TypeT> using VarDecl = TypeDecl<TypeT>;

//// **** Type Expressions ////

// *** Enum
struct Enum {
  std::vector<std::string> elems;
};

void to_json(nlohmann::json &j, const Enum &c);

// *** Union
struct Union {
  std::vector<std::string> elems;
};

void to_json(nlohmann::json &j, const Union &c);

// *** Record
struct Record {
  std::vector<std::pair<std::string, std::string>> elems;
};

void to_json(nlohmann::json &j, const Record &record);

struct RecordV2 {
  std::vector<nlohmann::json> decls;
};

void to_json(nlohmann::json &j, const RecordV2 &record);

// *** ScalarSet
template <class T>
struct is_string : public std::is_same<std::string, typename std::decay_t<T>> {
};

template <class T> class ScalarSet {
public:
  ScalarSet() = delete;
  explicit ScalarSet(T value) {
    static_assert(std::is_integral<T>() || is_string<T>::value,
                  "ScalarSet value must be integral or string");

    this->value = value;
  }
  T value;
};

void to_json(nlohmann::json &j, const ScalarSet<std::string> &ss);
void to_json(nlohmann::json &j, const ScalarSet<size_t> &ss);

// *** ID
struct ID {
  std::string id;
};
void to_json(nlohmann::json &j, const ID &id);

struct ExprID {
  std::string id;
};
void to_json(nlohmann::json &j, const ExprID &id);

// *** Array
template <class IndexT, class TypeT> struct Array {
  IndexT index;
  TypeT type;
};

template <class IndexT, class TypeT>
void to_json(nlohmann::json &j, const Array<IndexT, TypeT> &arr) {
  j = {{"typeId", "array"},
       {"type", {{"index", arr.index}, {"type", arr.type}}}};
}

// *** SubRange
template <class StartT, class StopT> struct SubRange {
  StartT start;
  StopT stop;
};

template <class StartT, class StopT>
void to_json(nlohmann::json &j, const SubRange<StartT, StopT> &sub_range) {
  j = {{"typeId", "sub_range"},
       {"type", {{"start", sub_range.start}, {"stop", sub_range.stop}}}};
}

// *** Multiset
template <class IndexT, class TypeT> struct Multiset {
  IndexT index;
  TypeT type;
};

template <class IndexT, class TypeT>
void to_json(nlohmann::json &j, const Multiset<IndexT, TypeT> &m) {
  j = {{"typeId", "multiset"},
       {"type", {{"index", m.index}, {"type", m.type}}}};
}

// *** Formal Parameter
template <class TypeT> struct Formal {
  std::string id;
  TypeT type;
  bool passByReference = false; // default pass-by-value
};

template <class TypeT>
void to_json(nlohmann::json &j, const Formal<TypeT> &formal) {
  j = {{"id", formal.id},
       {"type", formal.type},
       {"passByReference", formal.passByReference}};
}

// *** Forward Decls
template <class TypeT> struct ForwardDecl {
  std::string typeId;
  TypeT decl;
};

template <class TypeT>
void to_json(nlohmann::json &j, const ForwardDecl<TypeT> &fd) {
  j = {{"typeId", fd.typeId}, {"decl", fd.decl}};
}

struct Designator {
  std::string id;
  std::vector<nlohmann::json> indexes;
};

void to_json(nlohmann::json &j, const Designator &sd);

struct Indexer {
  std::string typeId;
  nlohmann::json index;
};

void to_json(nlohmann::json &j, const Indexer &indexer);

constexpr struct {
  const llvm::StringRef plus = "+";
  const llvm::StringRef minus = "-";
  const llvm::StringRef mult = "*";
  const llvm::StringRef div = "/";
  const llvm::StringRef logic_and = "&";
  const llvm::StringRef logic_or = "|";
  const llvm::StringRef logic_impl = "->";
  const llvm::StringRef less_than = "<";
  const llvm::StringRef less_than_eq = "<=";
  const llvm::StringRef grtr_than = ">";
  const llvm::StringRef grtr_than_eq = ">=";
  const llvm::StringRef eq = "=";
  const llvm::StringRef n_eq = "!=";
} BinaryOps;

template <class LHS, class RHS> struct BinaryExpr {
  LHS lhs;
  RHS rhs;
  llvm::StringRef
      op; // MUST BE ONE OF: +, -, *, /, &, |, &, ->, <, <=, >, >=, =, !=
};

template <class LHS, class RHS>
void to_json(nlohmann::json &j, const BinaryExpr<LHS, RHS> &binaryExpr) {
  j = {{"typeId", "binary"},
       {"expression",
        {{"lhs", binaryExpr.lhs},
         {"rhs", binaryExpr.rhs},
         {"op", binaryExpr.op}}}};
}

struct NegExpr {
  nlohmann::json expr;
};

void to_json(nlohmann::json &j, const NegExpr &negExpr);

// *** Statement Types

// TODO - designator is allowed to have 0 or more {rhs}
template <class T, class U> struct Assignment {
  T lhs;
  U rhs;
};

template <class T, class U>
void to_json(nlohmann::json &j, const Assignment<T, U> &a) {
  j = {{"typeId", "assignment"},
       {"statement", {{"lhs", a.lhs}, {"rhs", a.rhs}}}};
}

template <class T> struct Assert {
  T expr;
  std::string msg;
};

template <class T> void to_json(nlohmann::json &j, const Assert<T> &a) {
  j = {{"typeId", "assert"}, {"statement", {{"expr", a.expr}, {"msg", a.msg}}}};
}

template <class T> struct ForEachQuantifier {
  std::string id;
  T type;
};

template <class T>
void to_json(nlohmann::json &j, const ForEachQuantifier<T> &fe) {
  j = {{"typeId", "for_each"},
       {"quantifier", {{"id", fe.id}, {"type", fe.type}}}};
}

template <class StartIdx, class EndIdx> struct ForRangeQuantifier {
  std::string id;
  StartIdx start;
  EndIdx end;
};

template <class StartIdx, class EndIdx>
void to_json(nlohmann::json &j,
             const ForRangeQuantifier<StartIdx, EndIdx> &frq) {
  j = {
      {"typeId", "for_range"},
      {"quantifier", {{"id", frq.id}, {"start", frq.start}, {"end", frq.end}}}};
}

template <class T> struct ForStmt {
  T quantifier;
  std::vector<nlohmann::json> stmts;
};

template <class T> void to_json(nlohmann::json &j, const ForStmt<T> &forStmt) {
  j = {{"typeId", "for"},
       {"statement",
        {{"quantifier", forStmt.quantifier}, {"statements", forStmt.stmts}}}};
}

template <class T> struct IfStmt {
  T expr;
  std::vector<nlohmann::json> thenStmts;
  std::vector<nlohmann::json> elseStmts;
};

template <class T> void to_json(nlohmann::json &j, const IfStmt<T> &ifStmt) {
  j = {{"typeId", "if"},
       {"statement",
        {{"expr", ifStmt.expr}, {"thenStatements", ifStmt.thenStmts}}}};
  if (!ifStmt.elseStmts.empty()) {
    j["statement"]["elseStatements"] = ifStmt.elseStmts;
  }
}

template <class T> struct UndefineStmt { T value; };

template <class T> void to_json(nlohmann::json &j, const UndefineStmt<T> &uds) {
  j = {
      {"typeId", "undefine"},
      {"statement", {{"value", uds.value}}},
  };
}

template <class T> struct AliasStmt {
  std::string alias;
  T expr;
  std::vector<nlohmann::json> statements;
};

template <class T> void to_json(nlohmann::json &j, const AliasStmt<T> &as) {
  j = {{"typeId", "alias"},
       {"statement",
        {{"alias", as.alias},
         {"expr", as.expr},
         {"statements", as.statements}}}};
}

struct CaseStmt {
  nlohmann::json expr;
  std::vector<nlohmann::json> statements;
};

void to_json(nlohmann::json &j, const CaseStmt &caseStmt);

struct SwitchStmt {
  nlohmann::json expr;
  std::vector<CaseStmt> cases;
  std::vector<nlohmann::json> elseStatements;
};

void to_json(nlohmann::json &j, const SwitchStmt &sw);

struct ReturnStmt {
  nlohmann::json value;
};

void to_json(nlohmann::json &j, const ReturnStmt &rs);

struct MultisetCount {
  std::string varId;
  nlohmann::json varValue;
  nlohmann::json predicate;
};
void to_json(nlohmann::json &j, const MultisetCount &ms);

struct MultisetRemovePred {
  std::string varId;
  nlohmann::json varValue;
  nlohmann::json predicate;
};
void to_json(nlohmann::json &j, const MultisetRemovePred &msrp);

struct ProcCall {
  std::string funId;
  std::vector<nlohmann::json> actuals;
};

void to_json(nlohmann::json &j, const ProcCall &fn);

struct ProcCallExpr {
  std::string funId;
  std::vector<nlohmann::json> actuals;
};

void to_json(nlohmann::json &j, const ProcCallExpr &fn);

template <class QuantifierT> struct ForAll {
  QuantifierT quantifier;
  nlohmann::json expr;
};

struct ParensExpr {
  nlohmann::json expr;
};
void to_json(nlohmann::json &j, const ParensExpr &pExpr);

template <class QuantifierT>
void to_json(nlohmann::json &j, const ForAll<QuantifierT> &fa) {
  j = {{"typeId", "forall"},
       {"expression", {{"quantifier", fa.quantifier}, {"expr", fa.expr}}}};
}

/*
 * Rule States
 */

struct SimpleRule {
  std::string ruleDesc;
  nlohmann::json expr;
  std::vector<nlohmann::json> decls;
  std::vector<nlohmann::json> statements;
};

void to_json(nlohmann::json &j, const SimpleRule &sr);

struct RuleSet {
  std::vector<nlohmann::json> quantifiers;
  std::vector<nlohmann::json> rules;
};

void to_json(nlohmann::json &j, const RuleSet &rs);

struct AliasRule {
  std::string id;
  nlohmann::json expr;
  std::vector<nlohmann::json> rules;
};

void to_json(nlohmann::json &j, const AliasRule &ar);

struct ChooseRule {
  std::string index;
  nlohmann::json expr;
  std::vector<nlohmann::json> rules;
};

void to_json(nlohmann::json &j, const ChooseRule &cr);

struct StartState {
  std::string desc;
  std::vector<nlohmann::json> decls;
  std::vector<nlohmann::json> statements;
};

void to_json(nlohmann::json &j, const StartState &ss);

struct Invariant {
  std::string desc;
  nlohmann::json expr;
};

void to_json(nlohmann::json &j, const Invariant &inv);

/// ------------------------------------------------------------------------ ///
/// ***** SPECIFIC CASES *************************************************** ///
/// ------------------------------------------------------------------------ ///

/// -- Message Factory -- ///
struct MessageFactory {
  mlir::pcc::MsgDeclOp msgOp;
};
void to_json(nlohmann::json &j, const MessageFactory &mc);

/// --- Generic Murphi Function -- //
struct GenericMurphiFunction {
  std::string funcId;
  std::vector<nlohmann::json> params;
  nlohmann::json returnType;
  std::vector<nlohmann::json> forwardDecls;
  std::vector<nlohmann::json> statements;
};

void to_json(nlohmann::json &j, const GenericMurphiFunction &gmf);

/// --- Mutex Function --- ///
struct MutexFunction{
  bool is_acquire;
};

void to_json(nlohmann::json &j, const MutexFunction &mf);

/// -- Ordered Send Function -- ///
struct OrderedSendFunction {
  std::string netId;
};
void to_json(nlohmann::json &j, const OrderedSendFunction &sf);

/// -- Ordered Pop Function -- ///
struct OrderedPopFunction {
  std::string netId;
};
void to_json(nlohmann::json &j, const OrderedPopFunction &opf);

/// -- Unordered Send Function -- ///
struct UnorderedSendFunction {
  std::string netId;
};
void to_json(nlohmann::json &j, const UnorderedSendFunction &usf);

/// -- Machine Handler -- ///
struct MachineHandler {
  std::string machId;
  std::vector<nlohmann::json> cases;
};
void to_json(nlohmann::json &j, const MachineHandler &mh);

/// -- CPU Event Handler -- ///
struct CPUEventHandler {
  std::string start_state;
  std::vector<nlohmann::json> statements;
};
void to_json(nlohmann::json &j, const CPUEventHandler &cpuEventHandler);

/// -- CacheRuleHandler -- ///
struct CacheRuleHandler {
  std::vector<nlohmann::json> rules;
};
void to_json(nlohmann::json &j, const CacheRuleHandler &crh);

/// -- CPU Event Rule -- ///
struct CPUEventRule {
  std::string state;
  std::string event;
  bool atomic;
};
void to_json(nlohmann::json &j, const CPUEventRule &er);

/// -- Ordered Ruleset -- ///
struct OrderedRuleset {
  std::string netId;
};
void to_json(nlohmann::json &j, const OrderedRuleset &orderedRuleset);

/// -- UnorderedRuleset -- ///
struct UnorderedRuleset {
  std::string netId;
};
void to_json(nlohmann::json &j, const UnorderedRuleset &urs);

/// -- SWMR Invariant -- ///
struct SWMRInvariant {};
void to_json(nlohmann::json &j, const SWMRInvariant &swmrInv);

/// Used to represent a set Type;
struct Set{
    std::string elementType;
    size_t size;
    std::string getSetId() const;
    std::string getCntType() const;

    // set ops
    std::string set_add_fname() const;
    std::string set_count_fname() const;
    std::string set_contains_fname() const;
    std::string set_delete_fname() const;
    std::string set_clear_fname() const;
};

void to_json(nlohmann::json &j, const Set &theSet);

struct MulticastSend{
  std::string netId;
  Set theSet;
  std::string m_cast_fname() const;
};

void to_json(nlohmann::json &j, const MulticastSend &mcast);

/// -- Set Add -- ///
struct SetAdd {
  Set theSet;
};
void to_json(nlohmann::json &j, const SetAdd &setAdd);

/// -- Set Count -- ///
struct SetCount {
  Set theSet;
};
void to_json(nlohmann::json &j, const SetCount &setAdd);

/// -- Set Contains -- ///
struct SetContains {
  Set theSet;
};
void to_json(nlohmann::json &j, const SetContains &setAdd);

/// -- Set Delete -- ///
struct SetDelete {
  Set theSet;
};
void to_json(nlohmann::json &j, const SetDelete &setAdd);

/// -- Set Clear -- ///
struct SetClear {
  Set theSet;
};
void to_json(nlohmann::json &j, const SetClear &setAdd);

} // namespace detail
} // namespace murphi