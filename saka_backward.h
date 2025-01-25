#include <memory>
#include <stack>
#include <map>
#include <set>
#include <queue>
namespace saka
{
   template <class T>
   struct Pair
   {
       Pair(){}
       Pair(T l, T r) :lhs(l), rhs(r) {}
       T lhs, rhs;
   };

   class Node {
   public:
       virtual ~Node() {}
       virtual std::string nodeType() const = 0;
   };

   class Val;
   class Func : public Node
   {
   public:
       Pair<std::shared_ptr<Val>> inputs;

       virtual float forward(Pair<float> xs) const = 0;
       virtual Pair<float> backward(Pair<float> xs, float dy) const = 0;
   };
   class Val : public Node
   {
   public:
       float value;
       float derivative = 0.0f;
       int generation = 0;
       std::shared_ptr<Func> manufacturer;
       
       std::string nodeType() const { return "Value"; }
   };

   class ValRef
   {
   public:
       ValRef() {} // null
       ValRef(float value):m_impl(new Val())
       {
           m_impl->value = value;
       }
       void backward()
       {
           m_impl->derivative = 1.0f;

           auto generationOrder = [](std::shared_ptr<Val> lhs, std::shared_ptr<Val> rhs)
           {
               return lhs->generation < rhs->generation;
           };

           //std::stack<std::shared_ptr<Val>> stack;

           std::priority_queue<
               std::shared_ptr<Val>, 
               std::vector<std::shared_ptr<Val>>,
               decltype(generationOrder)
           > stack(generationOrder);
           std::set<std::shared_ptr<Val>> processed;

           stack.push(m_impl);

           

           while (!stack.empty())
           {
               std::shared_ptr<Val> val = stack.top(); stack.pop();
               if (!val->manufacturer)
               {
                   continue;
               }

               if (processed.count(val))
               {
                   continue;
               }
               processed.insert(val);

               // printf("g: %d, %s\n", val->generation, val->manufacturer->nodeType().c_str());

               Pair<std::shared_ptr<Val>> inputs = val->manufacturer->inputs;
               Pair<float> ds = val->manufacturer->backward({ inputs.lhs->value, inputs.rhs ? inputs.rhs->value : 0.0f }, val->derivative);
               inputs.lhs->derivative += ds.lhs;
               stack.push(inputs.lhs);

               if (inputs.rhs)
               {
                   inputs.rhs->derivative += ds.rhs;
                   stack.push(inputs.rhs);
               }
           }
       }
       float derivative() const {
           return m_impl->derivative;
       }
       std::string dotLang() const
       {
           std::string s;
           s += "digraph g{\n";

           std::stack<std::shared_ptr<Val>> stack;
           stack.push(m_impl);

           std::map<std::shared_ptr<Node>, int> idTable;

           int nodeIdx = 0;
           while (!stack.empty())
           {
               std::shared_ptr<Val> val = stack.top(); stack.pop();
               int valIdx = nodeIdx++;
               idTable[val] = valIdx;

               if (!val->manufacturer)
               {
                   continue;
               }

               int funcIdx = nodeIdx++;
               idTable[val->manufacturer] = funcIdx;

               stack.push(val->manufacturer->inputs.lhs);

               if (val->manufacturer->inputs.rhs)
               {
                   stack.push(val->manufacturer->inputs.rhs);
               }
           }

           for (auto nodes : idTable)
           {
               std::shared_ptr<Node> node = nodes.first;
               int idx = nodes.second;
               
               std::shared_ptr<Val> value = std::dynamic_pointer_cast<Val>(node);

               char label[256];
               if (value)
               {
                   sprintf(label, "%d [shape=record, label=\"{v:%.3f|d:%.3f|g:%d}}\"]\n", idx, value->value, value->derivative, value->generation);
               }
               else
               {
                   sprintf(label, "%d [shape=box, label=\"%s\",style=filled,color=lightblue]\n", idx, node->nodeType().c_str() );
               }
               
               s += label;
           }

           stack.push(m_impl);
           while (!stack.empty())
           {
               std::shared_ptr<Val> val = stack.top(); stack.pop();

               if (!val->manufacturer)
               {
                   continue;
               }
               char label[256];
               sprintf(label, "%d -> %d\n", idTable[val->manufacturer], idTable[val]);
               s += label;

               stack.push(val->manufacturer->inputs.lhs);

               sprintf(label, "%d -> %d\n", idTable[val->manufacturer->inputs.lhs], idTable[val->manufacturer]);
               s += label;

               if (val->manufacturer->inputs.rhs)
               {
                   sprintf(label, "%d -> %d\n", idTable[val->manufacturer->inputs.rhs], idTable[val->manufacturer]);
                   s += label;
                   stack.push(val->manufacturer->inputs.rhs);
               }
           }

           s += "}\n";

           return s;
       }
       std::shared_ptr<Val> m_impl;
   };
   class FuncRef
   {
   public:
       FuncRef(std::shared_ptr<Func> f):m_impl(f){}
       ValRef forward(Pair<ValRef> xs)
       {
           float y = m_impl->forward({ xs.lhs.m_impl->value, xs.rhs.m_impl ? xs.rhs.m_impl->value : 0.0f });
           ValRef r(y);
           m_impl->inputs = { xs.lhs.m_impl, xs.rhs.m_impl };
           r.m_impl->manufacturer = m_impl;

           int input_generation = xs.lhs.m_impl->generation;
           if (xs.rhs.m_impl)
           {
               input_generation = std::max(input_generation, xs.rhs.m_impl->generation);
           }
           r.m_impl->generation = input_generation + 1;
           return r;
       }
       std::shared_ptr<Func> m_impl;
   };

   class Square : public Func
   {
   public:
       virtual float forward(Pair<float> xs) const override
       {
           return xs.lhs * xs.lhs;
       }
       virtual Pair<float> backward(Pair<float> xs, float dy) const override
       {
           return Pair<float>( dy * 2.0f * xs.lhs, 0.0f );
       }
       std::string nodeType() const { return "Square"; }
   };
   class Exp : public Func
   {
   public:
       virtual float forward(Pair<float> xs) const override
       {
           return std::expf(xs.lhs);
       }
       virtual Pair<float> backward(Pair<float> xs, float dy) const override
       {
           return { dy * std::expf(xs.lhs), 0.0f };
       }
       std::string nodeType() const { return "Exp"; }
   };

   class Plus : public Func
   {
   public:
       virtual float forward(Pair<float> xs) const override
       {
           return xs.lhs + xs.rhs;
       }
       virtual Pair<float> backward(Pair<float> xs, float dy) const override
       {
           return { dy, dy };
       }
       std::string nodeType() const { return "Plus"; }
   };
   class Mul : public Func
   {
   public:
       virtual float forward(Pair<float> xs) const override
       {
           return xs.lhs * xs.rhs;
       }
       virtual Pair<float> backward(Pair<float> xs, float dy) const override
       {
           return { dy * xs.rhs, dy * xs.lhs };
       }
       std::string nodeType() const { return "Mul"; }
   };

   inline ValRef square(ValRef x)
   {
       FuncRef f(std::shared_ptr<Func>(new Square()));
       return f.forward({ x, ValRef()});
   }
   inline ValRef exp(ValRef x)
   {
       FuncRef f(std::shared_ptr<Func>(new Exp()));
       return f.forward({ x, ValRef() });
   }
   inline ValRef operator+(ValRef a, ValRef b)
   {
       FuncRef f(std::shared_ptr<Func>(new Plus()));
       return f.forward({ a, b });
   }
   inline ValRef operator*(ValRef a, ValRef b)
   {
       FuncRef f(std::shared_ptr<Func>(new Mul()));
       return f.forward({ a, b });
   }
}