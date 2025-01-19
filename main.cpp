﻿#include "pr.hpp"
#include <iostream>
#include <memory>

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

    class Val_;
    class Func_
    {
    public:
        Pair<std::shared_ptr<Val_>> inputs;

        virtual ~Func_() {}
        virtual float forward(Pair<float> xs) const = 0;
        virtual Pair<float> backward(Pair<float> xs, float dy) const = 0;
    };
    class Val_
    {
    public:
        float value;
        float derivative = 0.0f;
        std::shared_ptr<Func_> manufacturer;

        void backward(bool leaf)
        {
            if (!manufacturer)
            {
                return;
            }
            if (leaf)
            {
                derivative = 1.0f;
            }
            Pair<std::shared_ptr<Val_>> inputs = manufacturer->inputs;
            Pair<float> ds = manufacturer->backward({ inputs.lhs->value, inputs.rhs->value }, derivative);
            inputs.lhs->derivative += ds.lhs;
            inputs.rhs->derivative += ds.rhs;
            inputs.lhs->backward(false /*leaf*/);
            inputs.rhs->backward(false /*leaf*/);
        }
    };

    class Val
    {
    public:
        Val(float value):m_impl(new Val_())
        {
            m_impl->value = value;
        }
        void backward()
        {
            m_impl->backward(true /*leaf*/);
        }
        float derivative() const {
            return m_impl->derivative;
        }
        std::shared_ptr<Val_> m_impl;
    };
    class Func
    {
    public:
        Func(std::shared_ptr<Func_> f):m_impl(f){}
        Val forward(Pair<Val> xs)
        {
            
            float y = m_impl->forward({ xs.lhs.m_impl->value, xs.rhs.m_impl->value });
            Val r(y);
            m_impl->inputs = { xs.lhs.m_impl, xs.rhs.m_impl };
            r.m_impl->manufacturer = m_impl;
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
    };

    Val square(Val x)
    {
        Func f(std::shared_ptr<Func_>(new Square()));
        return f.forward({ x, Val(0.0f)});
    }
    Val exp(Val x)
    {
        Func f(std::shared_ptr<Func_>(new Exp()));
        return f.forward({ x, Val(0.0f) });
    }
    Val plus(Val a, Val b)
    {
        Func f(std::shared_ptr<Func_>(new Plus()));
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

    {
        using namespace saka;
        Val val(1.4);
        Val r = plus(square(val), square(val));
        r.backward();

        float d = val.derivative();
        printf("%f\n", d);
    }

    {
        var x = 1.4;
        auto f = [](var x) { return x * x + x * x; };
        var y = f(x);
        auto [ux] = derivatives(y, wrt(x));
        printf("%f\n", ux);
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
