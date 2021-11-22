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

type Statement = AssignmentStmt | AssertStmt | ForStmt | IfStmt | UndefineStmt;
type StatementType = 'assignment' | 'assert' | "for" | "if" | "undefine";

interface AssignmentStmt {
    lhs: Designator;
    rhs: ExpressionDescription;
}

interface AssertStmt {
    expr: ExpressionDescription;
    msg: string;
}

interface Quantifier{
    typeId: "for_each" | "for_range";
    quantifier: ForEachQuantifier
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
    value: Designator | DesignatorExpr;
}

interface Designator {
    objectId: ID;
    objType: "array" | "object";
    index: ExpressionDescription
}

// HACK - This is for the case when you want to chain a designator onto the result of a previous designator
interface DesignatorExpr {
    des: Designator;
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

type Expression = Designator | DesignatorExpr | ID | MultisetCount | ProcCall;
type ExpressionType = "designator" | "designator_expr" | "ID" | "binary" | "ms_count" | "proc_call";

interface ExpressionDescription {
    typeId: ExpressionType;
    expression: Expression;
}


interface StatementDescription {
    typeId: StatementType;
    statement: Statement;
}


interface Murphi_json_schema {
    decls: {
        const_decls?: ConstDecl[];
        type_decls?: TypeDecl[];
        var_decls?: VarDecl[];
    };
    proc_decls: ProcDecl[];

}