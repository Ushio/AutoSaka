#include "pr.hpp"
#include <iostream>
#include <memory>
#include <stack>
#include <map>
#include <set>
#include <queue>

#include "autodiff/reverse/var.hpp"

namespace saka
{
    template <class T>
    struct Pair
    {
        Pair(){}
        Pair(T l, T r) :lhs(l), rhs(r) {}
        T lhs, rhs;
    };

    class Node_ {
    public:
        virtual ~Node_() {}
        virtual std::string nodeType() const = 0;
    };

    class Val_;
    class Func_ : public Node_
    {
    public:
        Pair<std::shared_ptr<Val_>> inputs;

        virtual float forward(Pair<float> xs) const = 0;
        virtual Pair<float> backward(Pair<float> xs, float dy) const = 0;
    };
    class Val_ : public Node_
    {
    public:
        float value;
        float derivative = 0.0f;
        int generation = 0;
        std::shared_ptr<Func_> manufacturer;
        
        std::string nodeType() const { return "Value"; }
    };

    class Val
    {
    public:
        Val() {} // null
        Val(float value):m_impl(new Val_())
        {
            m_impl->value = value;
        }
        void backward()
        {
            m_impl->derivative = 1.0f;

            auto generationOrder = [](std::shared_ptr<Val_> lhs, std::shared_ptr<Val_> rhs)
            {
                return lhs->generation < rhs->generation;
            };

            //std::stack<std::shared_ptr<Val_>> stack;

            std::priority_queue<
                std::shared_ptr<Val_>, 
                std::vector<std::shared_ptr<Val_>>,
                decltype(generationOrder)
            > stack(generationOrder);
            std::set<std::shared_ptr<Val_>> processed;

            stack.push(m_impl);

            

            while (!stack.empty())
            {
                std::shared_ptr<Val_> val = stack.top(); stack.pop();
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

                Pair<std::shared_ptr<Val_>> inputs = val->manufacturer->inputs;
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

            std::stack<std::shared_ptr<Val_>> stack;
            stack.push(m_impl);

            std::map<std::shared_ptr<Node_>, int> idTable;

            int nodeIdx = 0;
            while (!stack.empty())
            {
                std::shared_ptr<Val_> val = stack.top(); stack.pop();
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
                std::shared_ptr<Node_> node = nodes.first;
                int idx = nodes.second;
                
                std::shared_ptr<Val_> value = std::dynamic_pointer_cast<Val_>(node);

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
                std::shared_ptr<Val_> val = stack.top(); stack.pop();

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
        std::shared_ptr<Val_> m_impl;
    };
    class Func
    {
    public:
        Func(std::shared_ptr<Func_> f):m_impl(f){}
        Val forward(Pair<Val> xs)
        {
            float y = m_impl->forward({ xs.lhs.m_impl->value, xs.rhs.m_impl ? xs.rhs.m_impl->value : 0.0f });
            Val r(y);
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
        std::shared_ptr<Func_> m_impl;
    };

    class Square : public Func_
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
    class Exp : public Func_
    {
    public:
        virtual float forward(Pair<float> xs) const override
        {
            return std::exp(xs.lhs);
        }
        virtual Pair<float> backward(Pair<float> xs, float dy) const override
        {
            return { dy * std::exp(xs.lhs), 0.0f };
        }
        std::string nodeType() const { return "Exp"; }
    };

    class Plus : public Func_
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
    class Mul : public Func_
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

    inline Val square(Val x)
    {
        Func f(std::shared_ptr<Func_>(new Square()));
        return f.forward({ x, Val()});
    }
    inline Val exp(Val x)
    {
        Func f(std::shared_ptr<Func_>(new Exp()));
        return f.forward({ x, Val() });
    }
    inline Val operator+(Val a, Val b)
    {
        Func f(std::shared_ptr<Func_>(new Plus()));
        return f.forward({ a, b });
    }
    inline Val operator*(Val a, Val b)
    {
        Func f(std::shared_ptr<Func_>(new Mul()));
        return f.forward({ a, b });
    }
}

using namespace autodiff;
//var f(var x, var y, var z)
//{
//    return (x + y + z) * exp(x * y * z);
//}
//var f(var x)
//{
//    return x * x;
//}
int main() {
    using namespace pr;

    //{
    //    using namespace saka;
    //    Val val(1.2);
    //    Val r = exp(square(val));
    //    r.backward();

    //    float d = val.derivative();
    //    printf("%f\n", d);
    //}

    //{
    //    var x = 1.2;
    //    auto f = [](var x) { return exp( x * x ); };
    //    var y = f(x);
    //    auto [ux] = derivatives(y, wrt(x)); 
    //    printf("%f\n", ux);
    //}

    //{
    //    using namespace saka;
    //    Val val(1.4);
    //    Val r = plus(square(val), square(val));
    //    r.backward();

    //    float d = val.derivative();
    //    printf("%f\n", d);
    //    printf("%s\n", r.dotLang().c_str());
    //}

    //{
    //    var x = 1.4;
    //    auto f = [](var x) { return x * x + x * x; };
    //    var y = f(x);
    //    auto [ux] = derivatives(y, wrt(x));
    //    printf("%f\n", ux);
    //}

    // A complex graph
    {
        using namespace saka;
        Val x(1.4f);
        Val a = square(x);
        //Val a = x;
        Val b = exp(a);
        Val c = square(a);
        Val y = b + c;
        
        y.backward();

        float d = x.derivative();
        printf("%f\n", d);

        printf("%s\n", y.dotLang().c_str());
    }

    {
        var x = 1.4f;
        auto f = [](var x) {
            auto a = x * x;
            //auto a = x;
            auto b = exp(a);
            auto c = a * a;
            return b + c;
        };
        var y = f(x);
        auto [ux] = derivatives(y, wrt(x));
        printf("%f\n", ux);
    }

    {
        var x = 1.4f;
        auto f = [](var x) {
            auto a = x * x;
            //auto a = x;
            auto b = exp(a);
            auto c = a * a;
            return saka::Pair<var>(b * c, b + c);
        };
        saka::Pair<var> y = f(x);
        auto [dydx0] = derivatives(y.lhs, wrt(x));
        auto [dydx1] = derivatives(y.rhs, wrt(x));
        printf("%f\n", dydx0);
        printf("%f\n", dydx1);
    }

    //var x = 1.0;         // the input variable x
    //var y = 2.0;         // the input variable y
    //var z = 3.0;         // the input variable z
    //var u = f(x, y, z);  // the output variable u

    //auto [ux, uy, uz] = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 4, 4, 4 };
    camera.lookat = { 0, 0, 0 };
    camera.zUp = true;

    double e = GetElapsedTime();

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        //DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        //DrawXYZAxis(1.0f);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
