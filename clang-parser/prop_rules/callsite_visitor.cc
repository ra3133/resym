#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Lex/Lexer.h"
#include <iostream>
#include "propagation_rule.hh"
using namespace clang;
using namespace std;


struct CallsiteInfo {
    string funName;
    vector<string> args;
    int lineNum; // Optional: Line number of the function call
};



CallsiteVisitor::CallsiteVisitor(ASTContext &C, Rewriter &R): 
            context(C), rewriter(R) {}

CallsiteVisitor::~CallsiteVisitor() {
    for (CallsiteInfo* info : callsitesInfo) {
        delete info;
    }
}


unsigned CallsiteVisitor::getLineNumber(Expr *expr) {
    FullSourceLoc FullLoc(expr->getBeginLoc(), context.getSourceManager());
    return FullLoc.getSpellingLineNumber();
}

bool CallsiteVisitor::VisitCallExpr(CallExpr *call) {
        if (FunctionDecl *FD = call->getDirectCallee()) {
            
            string funName = FD->getNameInfo().getName().getAsString();
            // llvm::outs() << "Function called: " << funName << "\n";
            
            // Check if the function name starts with "sub_"
            if (funName.rfind("sub_", 0) != 0) {
                // If not, skip this function call
                return true;
            }
            CallsiteInfo *info = new CallsiteInfo();

            info->funName = funName;
            info->lineNum = getLineNumber(call);

            for (unsigned i = 0, e = call->getNumArgs(); i != e; ++i) {
                if (call->getArg(i)) {
                    std::string argText = Lexer::getSourceText(CharSourceRange::getTokenRange(call->getArg(i)->getSourceRange()), context.getSourceManager(), context.getLangOpts()).str();
                    // llvm::outs() << "Arg " << i << ": " << argText << "\n";
                    info->args.push_back(argText);
                }
            }
            callsitesInfo.push_back(info);
        }
        return true;
    }


// Define how to convert a MemoryAccessInfo object to JSON
void to_json(nlohmann::json &j, const CallsiteInfo &info) {
    j = nlohmann::json{
        {"funName", info.funName},
        {"args", info.args},
        {"lineNum", info.lineNum},
    };
}


nlohmann::json CallsiteVisitor::dumpCallsitesToJson(){
    nlohmann::json j;
    // Convert each MemoryAccessInfo object in the memberAccess vector to JSON
    for (const auto &info : callsitesInfo) {
        j.push_back(*info);
    }

    return j;
}