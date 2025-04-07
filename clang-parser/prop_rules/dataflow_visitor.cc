#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Lex/Lexer.h"
#include <iostream>
#include "propagation_rule.hh"
using namespace clang;
using namespace std;



struct VariableAssignmentInfo {
    std::string lhsVarName;  // Left-hand side variable name
    std::string rhsVarName;  // Right-hand side variable name
    int lineNum;             // Line number of the assignment
};


DataflowVisitor::DataflowVisitor(ASTContext &C, Rewriter &R): 
            context(C), rewriter(R) {}

DataflowVisitor::~DataflowVisitor() {
    for (VariableAssignmentInfo* info : assignments) {
        delete info;
    }
}


unsigned DataflowVisitor::getLineNumber(Expr *expr) {
    FullSourceLoc FullLoc(expr->getBeginLoc(), context.getSourceManager());
    return FullLoc.getSpellingLineNumber();
}

bool DataflowVisitor::isVariableOrArgument(DeclRefExpr *expr) {
    if (auto *varDecl = dyn_cast<VarDecl>(expr->getDecl())) {
        return varDecl->isLocalVarDecl() || isa<ParmVarDecl>(varDecl);
    }
    return false;
}


bool DataflowVisitor::VisitBinaryOperator(BinaryOperator *bo) {
    if (bo->isAssignmentOp()) {
        if (DeclRefExpr *lhs = dyn_cast<DeclRefExpr>(bo->getLHS()->IgnoreImpCasts())) {
            if (DeclRefExpr *rhs = dyn_cast<DeclRefExpr>(bo->getRHS()->IgnoreImpCasts())) {
                // Check if both LHS and RHS are variables or arguments
                if (isVariableOrArgument(lhs) && isVariableOrArgument(rhs)) {
                    VariableAssignmentInfo* info = new VariableAssignmentInfo();
                    info->lhsVarName = lhs->getNameInfo().getAsString();
                    info->rhsVarName = rhs->getNameInfo().getAsString();
                    // info.lineNum = context.getSourceManager().getSpellingLineNumber(bo->getLocStart());
                    info->lineNum = getLineNumber(bo);
                    // cout << info->lhsVarName << " = " << info->rhsVarName << " (line " << info->lineNum << ")\n";
                    assignments.push_back(info);
                }
            }
        }
    }
    return true;
}


// Define how to convert a MemoryAccessInfo object to JSON
void to_json(nlohmann::json &j, const VariableAssignmentInfo &info) {
    j = nlohmann::json{
        {"lhsVarName", info.lhsVarName},
        {"rhsVarName", info.rhsVarName},
        {"lineNum", info.lineNum},
    };
}


nlohmann::json DataflowVisitor::dumpDataflowToJson(){
    nlohmann::json j;
    // Convert each MemoryAccessInfo object in the memberAccess vector to JSON
    for (const auto &info : assignments) {
        j.push_back(*info);
    }

    return j;
}