#ifndef CONSUMER_HH
#define CONSUMER_HH

#include "configUtils.hh"
#include "clang/AST/ASTConsumer.h"
#include "propagation_rule.hh"
// #include <nlohmann/json.hpp>

using namespace clang;
using namespace std;

class MyConsumer : public ASTConsumer {
public:
  explicit MyConsumer(ASTContext &C, Rewriter &R) : context(C), rewriter(R) {}

protected:
  Rewriter &rewriter;
  ASTContext &context;
};


class MyFieldAccessConsumer : public MyConsumer {
// constructor
public:
    explicit MyFieldAccessConsumer(ASTContext &C, Rewriter &R)
    : MyConsumer(C, R), fieldAccessVisitor(C, R), context(C), rewriter(R) {
    }

    virtual void HandleTranslationUnit(ASTContext &context){
        fieldAccessVisitor.TraverseDecl(context.getTranslationUnitDecl());
    }
    
    void dumpAccessToJson(const string &filename){
      nlohmann::json j = fieldAccessVisitor.dumpAccessToJson();
      if(!j.empty()){
        writeJSONToFile(j, filename);
      }
      else{
        DBG_OUT << "JSON data is empty. Skipping file write." <<endl;
      }
      
    }

private:
    Rewriter &rewriter;
    ASTContext &context;
    FieldAccessVisitor fieldAccessVisitor;
};



class MyCallsiteConsumer : public MyConsumer {
// constructor
public:
    explicit MyCallsiteConsumer(ASTContext &C, Rewriter &R)
    : MyConsumer(C, R), callsiteVisitor(C, R), context(C), rewriter(R) {}

    virtual void HandleTranslationUnit(ASTContext &context){
        callsiteVisitor.TraverseDecl(context.getTranslationUnitDecl());
    }
    
    void dumpCallsitesToJson(const string &filename){
      nlohmann::json j = callsiteVisitor.dumpCallsitesToJson();
      if(!j.empty()){
        writeJSONToFile(j, filename);
      }
      else{
        DBG_OUT << "JSON data is empty. Skipping file write." <<endl;
      }
      
    }

private:
    Rewriter &rewriter;
    ASTContext &context;
    CallsiteVisitor callsiteVisitor;
};



class MyDataflowConsumer : public MyConsumer {
// constructor
public:
    explicit MyDataflowConsumer(ASTContext &C, Rewriter &R)
    : MyConsumer(C, R), dataflowVisitor(C, R), context(C), rewriter(R) {}

    virtual void HandleTranslationUnit(ASTContext &context){
        dataflowVisitor.TraverseDecl(context.getTranslationUnitDecl());
    }
    
    void dumpDataflowToJson(const string &filename){
      nlohmann::json j = dataflowVisitor.dumpDataflowToJson();
      if(!j.empty()){
        writeJSONToFile(j, filename);
      }
      else{
        DBG_OUT << "JSON data is empty. Skipping file write." <<endl;
      }
      
    }

private:
    Rewriter &rewriter;
    ASTContext &context;
    DataflowVisitor dataflowVisitor;
};




#endif
