// THIS IS USED AS A REFERENCE/DOCUMENTATION
// FOR THE JSON OBJECT THAT WILL BE GENERATED

// ***** DECLS ***** //

interface TypeDescription {
    typeId: MurphiTypeId;
    type: MurphiType;
}

interface ConstDecl {
    id: string;
    value: number;
}

interface TypeDecl extends TypeDescription {
    id: string;
}

interface VarDecl extends TypeDescription {
    id: string;
}

type Decl = ConstDecl | TypeDecl | VarDecl;

// ***** TYPES ***** //

type MurphiTypeId = "record" | "enum" | "sub_range" | "array" | "ID" | "multiset" | "scalarset" | "union";

interface MurphiRecord {
    decls:
        {
            id: string;
            typeId: MurphiTypeId;
            type: MurphiType;
        }[];
}

interface Enum {
    decls: ID[];
}

// TODO - subranges should support expr for start/stop
interface IntegerSubRange {
    start: number | string;
    stop: number | string;
}

interface MurphiArray {
    index: TypeDescription;
    type: TypeDescription;
}

interface MultisetType {
    index: TypeDescription;
    type: TypeDescription;
}

interface ScalarsetType {
    type: ID | number;
}

type ArrayTwoOrMore<T> = [T, T, ...T[]];

interface UnionType {
    listElems: ArrayTwoOrMore<ID>
}

type ID = string;
type MurphiType = MurphiRecord | Enum | IntegerSubRange | MurphiArray | ID | MultisetType | ScalarsetType | UnionType;


// ***** METHODS ***** //

interface IdAndType {
    id: string;
    typeId: MurphiTypeId;
    type: MurphiType;
}

interface ProcDecl {
    procType: ProcType;
    def: ProcDef;
}

type ProcType = "procedure" | "function";
type ProcDef = MurphiFunction | MurphiProcedure;

interface MurphiFunction {
    id: ID;
    params: Formal[];
    returnType: TypeDescription;
    forwardDecls?: FwdDecl[];
    statements?: StatementDescription[]
}

interface MurphiProcedure {
    id: ID;
    params: Formal[];
    forwardDecls?: FwdDecl[];
    statements?: StatementDescription[];
}

interface Formal {
    id: ID;
    type: TypeDescription;
}

interface FwdDecl {
    typeId: "const" | "var" | "type";
    decl: TypeDecl | VarDecl | ConstDecl;
}

type Statement = AssignmentStmt | AssertStmt | ForStmt | IfStmt | UndefineStmt | ProcCall | AliasStmt | SwitchStmt;
type StatementType = 'assignment' | 'assert' | "for" | "if" | "undefine" | "proc_call" | "alias" | "switch";

interface AssignmentStmt {
    lhs: DesignatorDescription;
    rhs: ExpressionDescription;
}

class DesignatorDescription implements ExpressionDescription {
    typeId: "designator" | "designator_expr";
    expression: Designator | DesignatorExpr;
}

interface AssertStmt {
    expr: ExpressionDescription;
    msg: string;
}


interface AliasStmt {
    alias: ID;
    expr: ExpressionDescription;
    statements: StatementDescription[];
}

interface CaseStmt{
    expr: ExpressionDescription;
    statements: StatementDescription[];
}

interface SwitchStmt{
    expr: ExpressionDescription;
    cases: CaseStmt[];
    elseStatements: StatementDescription[];
}

interface Quantifier{
    typeId: "for_each" | "for_range";
    quantifier: ForEachQuantifier | ForRangeQuantifier
}

interface ForEachQuantifier{
    id: ID;
    type: TypeDescription;
}

interface ForRangeQuantifier{
    id: ID;
    start: ExpressionDescription;
    end: ExpressionDescription;
}

interface ForStmt{
    quantifier: Quantifier;
    statements: StatementDescription[];
}

interface IfStmt{
    expr: ExpressionDescription;
    thenStatements: StatementDescription[];
    elseStatements?: StatementDescription[];
}

interface UndefineStmt{
    value: DesignatorDescription;
}

interface Designator {
    objId: ID;
    objType: "array" | "object";
    index: ExpressionDescription
}

// HACK - This is for the case when you want to chain a designator onto the result of a previous designator
interface DesignatorExpr {
    des: DesignatorDescription;
    objType: "array" | "object";
    index: ExpressionDescription;
}

interface BinaryExpr {
    lhs: ExpressionDescription;
    rhs: ExpressionDescription;
    op: BinaryOp;
}

interface MultisetCount{
    varId: ID;
    varValue: ExpressionDescription;
    predicate: ExpressionDescription;
}

interface ProcCall{
    funId: ID;
    actuals: ExpressionDescription[]
}

type BinaryOp = '+' | '-' | '*' | '/' | '&' | '->' | '<' | '<=' | '>' | '>=' | '=' | '!=';

type Expression = Designator | DesignatorExpr | ID | MultisetCount | BinaryExpr;
type ExpressionType = "designator" | "designator_expr" | "ID" | "binary" | "ms_count";

interface ExpressionDescription {
    typeId: ExpressionType;
    expression: Expression;
}


interface StatementDescription {
    typeId: StatementType;
    statement: Statement;
}

/*
Rules
 */

type Rule = SimpleRule | RuleSet | AliasRule;
type RuleType = "simple_rule" | "ruleset" | "alias_rule";

interface RuleDescription{
    typeId: RuleType;
    rule: Rule;
}

interface SimpleRule{
    ruleDesc: string;
    expr: ExpressionDescription;
    decls?: FwdDecl[];
    statements: StatementDescription[];
}

interface RuleSet{
    quantifiers: Quantifier[];
    rules: RuleDescription[];
}

interface AliasRule{
    id: ID;
    expr: ExpressionDescription;
    rules: RuleDescription[];
}


interface Murphi_json_schema {
    decls: {
        const_decls?: ConstDecl[];
        type_decls?: TypeDecl[];
        var_decls?: VarDecl[];
    };
    proc_decls: ProcDecl[];
    rules: RuleDescription[];
}