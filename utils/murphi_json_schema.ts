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

interface TypeDecl extends TypeDescription{
    id: string;
}

interface VarDecl extends TypeDescription{
    id: string;
}

type Decl = ConstDecl | TypeDecl | VarDecl;

// ***** TYPES ***** //

type MurphiTypeId = "record" | "enum" | "sub_range" | "array" | "ID" | "multiset" | "scalarset";
interface MurphiRecord {
    decls:[
        {
            id: string;
            typeId: MurphiTypeId;
            type: MurphiType;
        }
    ];
}

interface Enum {
    decls:ID[];
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
    type: string;
}

type ID = string;
type MurphiType = MurphiRecord | Enum | IntegerSubRange | MurphiArray | ID | MultisetType | ScalarsetType;


// ***** METHODS ***** //

interface IdAndType {
    id: string;
    typeId: MurphiTypeId;
    type: MurphiType;
}


interface Murphi_json_schema {
    decls: {
        const_decls?: ConstDecl[];
        type_decls?: TypeDecl[];
        var_decls?: VarDecl[];
    };
}