grammar ProtoCC;

options
{
    language=Cpp;
}

NETWORK : 'Network';
CACHE : 'Cache';
DIR : 'Directory';
MEM : 'Memory';

MSG : 'Message';

STATE : 'State';
DATA : 'Data';
NID : 'ID';
FIFO : 'FIFO';

ARCH : 'Architecture';
PROC : 'Process';
STABLE : 'Stable';

DOT : '.';
COMMA : ',';
DDOT : ':';
SEMICOLON : ';';
EQUALSIGN : '=';
PLUS : '+';
MINUS : '-';
MULT : '*';

CONSTANT : '#';
BOOLID : 'bool';
INTID : 'int';
ARRAY : 'array';

SET: 'set';

AWAIT : 'await';
NEXT : 'next';
WHEN : 'when';
BREAK : 'break';

IF : 'if';
ELSE : 'else';

OCBRACE : '{';
CCBRACE : '}';
OEBRACE : '[';
CEBRACE : ']';
OBRACE : '(';
CBRACE : ')';

NEG : '!';

send_function
	: 'send'
	| 'Send'
	;

mcast_function
	: 'mcast'
	| 'Mcast'
	;

bcast_function
	: 'bcast'
	| 'Bcast'
	;

internal_event_function
	: 'Event'
	;

set_function_types
	: 'add'
	| 'count'
	| 'contains'
	| 'del'
	| 'clear'
	;

relational_operator
	: '=='
	| '!='
	| '<='
	| '>='
	| '<'
	| '>'
	;

combinatorial_operator
    : '&'
    | '|'
    ;

WS  :  [ \t\r\f\n]+ -> skip ;

COMMENT
    :   '/*' ()*? '*/' 
    ;

LineComment
  : '//' ~[\r\n]* -> channel(HIDDEN)
  ;

NEWLINE : '\r'? '\n' -> skip;

/** DataTypes */
BOOL:   'true' | 'false';

INT :	'0'..'9'+;

ACCESS: 'load' | 'store';
EVICT: 'evict';

ID  :	('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')* ;

/** Specifiers */
document
	:	(const_decl | init_hw | arch_block | expressions)*;


/** Var Decl */
declarations : int_decl | bool_decl | state_decl | data_decl | id_decl;

    const_decl : CONSTANT ID INT;

    int_decl : INTID range ID (EQUALSIGN INT)* SEMICOLON;
    bool_decl : BOOLID ID (EQUALSIGN BOOL)* SEMICOLON;

    state_decl : STATE ID SEMICOLON;
    data_decl : DATA ID SEMICOLON;
    id_decl : set_decl* NID ID (EQUALSIGN set_decl* ID)* SEMICOLON;

        set_decl : SET OEBRACE val_range CEBRACE;
        range : OEBRACE val_range DOT DOT val_range CEBRACE;
        val_range : INT | ID;

    array_decl : ARRAY range;
    fifo_decl: FIFO range;

/** Decl Static Objects */
init_hw : network_block | machines | message_block;
    object_block : object_expr SEMICOLON;
    object_expr : object_id | object_func;
    object_id:  ID;
    object_func : ID DOT object_idres (OBRACE object_expr* (COMMA object_expr)* CBRACE)*;
    object_idres: ID | NID;

    /** Machine */
    machines : cache_block | dir_block | mem_block;
        cache_block : CACHE OCBRACE declarations* CCBRACE objset_decl* ID SEMICOLON;
        dir_block : DIR OCBRACE declarations* CCBRACE objset_decl* ID SEMICOLON;
        mem_block : MEM OCBRACE declarations* CCBRACE objset_decl* ID SEMICOLON;

        objset_decl : SET OEBRACE val_range CEBRACE;

    /** Network */
    network_block : NETWORK OCBRACE network_element* CCBRACE SEMICOLON;
        element_type : 'Ordered' | 'Unordered';
        network_element : element_type ID SEMICOLON;
    network_send : ID DOT send_function OBRACE ID CBRACE SEMICOLON;
    network_bcast: ID DOT bcast_function OBRACE ID CBRACE SEMICOLON;
    network_mcast: ID DOT mcast_function OBRACE ID COMMA ID CBRACE SEMICOLON;

    /** Message */
    message_block : MSG ID OCBRACE declarations* CCBRACE SEMICOLON ;
    message_constr : ID OBRACE message_expr* (COMMA message_expr)* CBRACE ;
    message_expr : object_expr | set_func | INT | BOOL | NID;

    /** Set Functions */
    set_block : set_func SEMICOLON;
    set_func : ID DOT set_function_types OBRACE set_nest* CBRACE;
    set_nest : set_func | object_expr;

    /** Modify State Functions */

    internal_event_block: internal_event_func SEMICOLON;
    internal_event_func: internal_event_function OBRACE ID CBRACE;

/** Behavioural */
arch_block : ARCH ID OCBRACE arch_body CCBRACE;

arch_body: stable_def process_block*;

stable_def : STABLE OCBRACE ID (COMMA ID)* CCBRACE;

process_block : PROC process_trans OCBRACE process_expr* CCBRACE;
    process_trans : OBRACE ID COMMA process_events process_finalstate? CBRACE;
    process_finalstate: COMMA process_finalident;
    process_finalident: (ID | STATE);
    process_events : (ACCESS | EVICT | ID);
    process_expr: expressions | network_send | network_mcast | network_bcast |transaction;

/** TRANSACTIONS */
transaction : AWAIT OCBRACE trans* CCBRACE;
    trans : WHEN ID DDOT trans_body* ;
        trans_body : expressions | next_trans | next_break | transaction | network_send | network_mcast | network_bcast;
            next_trans: NEXT OCBRACE trans* CCBRACE ;

next_break : BREAK SEMICOLON;

/** Expressions */
expressions : assignment | conditional | object_block | set_block | internal_event_block;
assignment : process_finalident EQUALSIGN assign_types SEMICOLON;
    assign_types : object_expr | message_constr | math_op | set_func | INT | BOOL;
    math_op : val_range (PLUS | MINUS) val_range;

/** Conditional */
conditional: if_stmt | ifnot_stmt;
    if_stmt 
    : IF cond_comb OCBRACE if_expression CCBRACE
    (ELSE OCBRACE else_expression CCBRACE)*
    ;

    ifnot_stmt 
    : IF NEG cond_comb OCBRACE if_expression CCBRACE
    (ELSE  OCBRACE else_expression CCBRACE)*
    ;

        if_expression: exprwbreak*;
        else_expression: exprwbreak*;
        exprwbreak: expressions | network_send | network_mcast | network_bcast | transaction | next_break;
        cond_comb: cond_rel (combinatorial_operator cond_rel)*;
        cond_rel : OBRACE* cond_sel CBRACE*;
            cond_sel : cond_type_expr (relational_operator cond_type_expr)*;
            cond_type_expr: cond_types (indv_math_op cond_types)*;
            indv_math_op: (PLUS | MINUS | MULT);
            cond_types : object_expr | set_func | INT | BOOL | NID;

