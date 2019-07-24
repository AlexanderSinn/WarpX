#ifndef WP_PARSER_Y_H_
#define WP_PARSER_Y_H_

#ifdef __cplusplus
#include <cstdlib>
extern "C" {
#else
#include <stdlib.h>
#endif

enum wp_f1_t {  // Bulit-in functions with one argument
    WP_SQRT = 1,
    WP_EXP,
    WP_LOG,
    WP_LOG10,
    WP_SIN,
    WP_COS,
    WP_TAN,
    WP_ASIN,
    WP_ACOS,
    WP_ATAN,
    WP_SINH,
    WP_COSH,
    WP_TANH,
    WP_ABS,
    WP_POW_M3,
    WP_POW_M2,
    WP_POW_M1,
    WP_POW_P1,
    WP_POW_P2,
    WP_POW_P3
};

enum wp_f2_t {  // Built-in functions with two arguments
    WP_POW = 1,
    WP_GT,
    WP_LT,
    WP_GEQ,
    WP_LEQ,
    WP_EQ,
    WP_NEQ,
    WP_AND,
    WP_OR,
    WP_HEAVISIDE,
    WP_MIN,
    WP_MAX
};

enum wp_node_t {
    WP_NUMBER = 1,
    WP_SYMBOL,
    WP_ADD,
    WP_SUB,
    WP_MUL,
    WP_DIV,
    WP_NEG,
    WP_F1,
    WP_F2,
    WP_ADD_VP,  /* types below are generated by optimization */
    WP_ADD_PP,
    WP_SUB_VP,
    WP_SUB_PP,
    WP_MUL_VP,
    WP_MUL_PP,
    WP_DIV_VP,
    WP_DIV_PP,
    WP_NEG_P
};

/* In C, the address of the first member of a struct is the same as
 * the address of the struct itself.  Because of this, all struct wp_*
 * pointers can be passed around as struct wp_node pointer and enum
 * wp_node_t type can be safely checked to determine their real type.
 */

union wp_vp {
    double  v;
    double* p;
};

struct wp_node {
    enum wp_node_t type;
    struct wp_node* l;
    struct wp_node* r;
    union wp_vp lvp;  // After optimization, this may store left value/pointer.
    double* rp;       //                     this may store right      pointer.
};

struct wp_number {
    enum wp_node_t type;
    double value;
};

struct wp_symbol {
    enum wp_node_t type;
    char* name;
    double* pointer;
};

struct wp_f1 {  /* Builtin functions with one argument */
    enum wp_node_t type;
    struct wp_node* l;
    enum wp_f1_t ftype;
};

struct wp_f2 {  /* Builtin functions with two arguments */
    enum wp_node_t type;
    struct wp_node* l;
    struct wp_node* r;
    enum wp_f2_t ftype;
};

/*******************************************************************/

/* These functions are used in bison rules to generate the original
 * AST. */
void wp_defexpr (struct wp_node* body);
struct wp_node* wp_newnumber (double d);
struct wp_symbol* wp_makesymbol (char* name);
struct wp_node* wp_newsymbol (struct wp_symbol* sym);
struct wp_node* wp_newnode (enum wp_node_t type, struct wp_node* l,
                            struct wp_node* r);
struct wp_node* wp_newf1 (enum wp_f1_t ftype, struct wp_node* l);
struct wp_node* wp_newf2 (enum wp_f2_t ftype, struct wp_node* l,
                          struct wp_node* r);

void yyerror (char const *s, ...);

/*******************************************************************/

/* This is our struct for storing AST in a more packed way.  The whole
 * tree is stored in a contiguous chunk of memory starting from void*
 * p_root with a size of sz_mempool.
 */
struct wp_parser {
    void* p_root;
    void* p_free;
    struct wp_node* ast;
    size_t sz_mempool;
};

struct wp_parser* wp_parser_new (void);
void wp_parser_delete (struct wp_parser* parser);

struct wp_parser* wp_parser_dup (struct wp_parser* source);
struct wp_node* wp_parser_ast_dup (struct wp_parser* parser, struct wp_node* src, int move);

void wp_parser_regvar (struct wp_parser* parser, char const* name, double* p);
void wp_parser_setconst (struct wp_parser* parser, char const* name, double c);

/* We need to walk the tree in these functions */
void wp_ast_optimize (struct wp_node* node);
size_t wp_ast_size (struct wp_node* node);
void wp_ast_print (struct wp_node* node);
void wp_ast_regvar (struct wp_node* node, char const* name, double* p);
void wp_ast_setconst (struct wp_node* node, char const* name, double c);

double wp_call_f1 (enum wp_f1_t type, double a);
double wp_call_f2 (enum wp_f2_t type, double a, double b);

#ifdef __cplusplus
}
#endif

#endif
