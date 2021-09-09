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

type MurphiTypeId = "record" | "enum" | "sub_range" | "array" | "ID";
interface Record {
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
    start: number;
    stop: number;
}

interface MurphiArray {
    index: TypeDescription;
    type: TypeDescription;
}

type ID = string;
type MurphiType = Record | Enum | IntegerSubRange | MurphiArray | ID;


// ***** METHODS ***** //

type MurphiMethods = undefined;

interface IdAndType {
    id: string;
    typeId: MurphiTypeId;
    type: MurphiType;
}

interface GlobalMessageType {
    fields: IdAndType[]
}

interface MessageConstructor {
    msgId: string;
    additionalParameters: IdAndType[]
}



interface Murphi_json_schema {
    decls: {
        const_decls?: ConstDecl[];
        type_decls?: TypeDecl[];
        var_decls?: VarDecl[];
    };
    global_message_type: GlobalMessageType;
}