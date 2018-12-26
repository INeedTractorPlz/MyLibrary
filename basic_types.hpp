#define TYPES

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/lexical_cast.hpp>

#include <getopt.h>

#include <functional>
#include<vector>
#include<iterator>
#include<cmath>
#include<tuple>
#include <type_traits>
#define sign(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) )

#include<iostream>

/*#ifndef FUNCTIONS
#include"functions.hpp"
#endif
*/
using namespace boost::numeric::ublas;


typedef double data_type;
typedef vector<data_type> state_vector;
typedef matrix<data_type> state_matrix;

template<typename type>
inline std::ostream& operator<<(std::ostream& os, const std::vector<type>& in){
    for(auto& it : in)
        os << it << " ";
    return os;
}


template<typename ... Types>
inline void simple_cout(Types&&... args){
    (std::cout << ... << args) << std::endl;
}

template<typename type>
void fun_for_parser(type& parameter, char* optarg){
        std::string ss=boost::lexical_cast<std::string>(optarg);
        parameter = boost::lexical_cast<typename std::remove_reference<type>::type>(ss);
}
template <class T>
constexpr T comp_w_zero(T arg) {
  return arg > 0 ? arg : 0;
}

template<int number_shorts, int number_longs, unsigned summ, typename Head, typename... Args>
struct parser
{
    static void parse(std::tuple<Head, Args ...> parameters, int opt, char *optarg, 
    const std::vector<char>& short_keys, struct option* longOpts, int longIndex)
    {
       constexpr int l = comp_w_zero(sizeof...(Args) + 1 - summ);
        //Количество функций оставшихся в аргументах: кол-во аргументов
        //минус кол-во переменных. Когда функции кончаются, нужно просто приравнять к нулю.
        constexpr int k = summ - number_longs - number_shorts; //Кол-во обработанных перменных.
        constexpr int N = l + k; //Номер переменной, которую нужно обработать.
        auto beheading = [](Head head, Args... args){
            return std::make_tuple(std::ref(args)...);
        };
        std::tuple<Args...> parameters_without_Head = std::apply(beheading, parameters);
        if(opt == short_keys[k])//Переменную обрабатываем только если ключ соответствующий.
            if constexpr (l > 0)
                std::get<0>(parameters)(std::get<N>(parameters), optarg);
            else{
                typedef std::tuple<Head, Args...> MyTuple;
                fun_for_parser<std::tuple_element_t<N, MyTuple> >(std::get<N>(parameters), optarg);
                }
        if constexpr (sizeof...(Args) > 0)
            parser<number_shorts-1, number_longs, summ, Args...>::parse(parameters_without_Head, 
            opt, optarg, short_keys,longOpts, longIndex);
    }
};

template<int number_longs, unsigned summ, typename Head, typename... Args>
struct parser<0, number_longs, summ, Head, Args...>
{
    static void parse(std::tuple<Head, Args ...> parameters, int opt, char *optarg, 
    const std::vector<char>& short_keys,  struct option* longOpts, int longIndex)
    {
       constexpr int l = comp_w_zero(sizeof...(Args) + 1 - summ);
       constexpr int k = summ - number_longs;
        constexpr int N = l + k;
        auto beheading = [](Head head, Args... args){
            return std::make_tuple(std::ref(args)...);
        };
        std::tuple<Args...> parameters_without_Head = std::apply(beheading, parameters);
        if(opt == 0)
            if(longOpts[k - short_keys.size()].name == longOpts[longIndex].name)
                if constexpr (l > 0)
                    std::get<0>(parameters)(std::get<N>(parameters), optarg);
                else{
                typedef std::tuple<Head, Args...> MyTuple;
                fun_for_parser<std::tuple_element_t<N, MyTuple> >(std::get<N>(parameters), optarg);
                }   
        if constexpr (sizeof...(Args) > 0)
            parser<0,number_longs - 1, summ, Args...>::parse(parameters_without_Head, 
            opt, optarg, short_keys,longOpts, longIndex);
    }
};

template<unsigned summ, typename Head, typename... Args>
struct parser<0, 0, summ, Head, Args...>
{
    static void parse(std::tuple<Head, Args ...> parameters, int opt, char *optarg, 
    const std::vector<char>& short_keys,  struct option* longOpts, int longIndex)
    {    }
};

template<int number_shorts, int number_longs>
struct Parser_t{
    const char* short_opts;
    std::vector<char> short_keys;
    struct option* longOpts;
    Parser_t(const std::string& short_opts, struct option* longOpts = 0) :
    short_opts(short_opts.c_str()), longOpts(longOpts){
        for(unsigned i = 0; i < short_opts.size(); i += 2){
            short_keys.push_back(short_opts[i]);
        }
    }
    template<typename ... Types>
    void Parser(int argc, char *const argv[], Types&&... Args){
        std::tuple<Types&& ...> parameters = std::make_tuple(std::ref(Args)...);
        int longIndex; 
        auto opt =  getopt_long(argc,argv,short_opts,longOpts,&longIndex);
        while(opt != -1){
            constexpr unsigned summ = number_longs + number_shorts;
            parser<number_shorts, number_longs, summ, Types&&...>::parse(parameters,
            opt, optarg, short_keys,longOpts,longIndex);  
            opt = getopt_long(argc,argv,short_opts,longOpts,&longIndex);  
        }
    }
};

template<typename type>
struct Jacoby_t{
    //const state_matrix& input_matrix;
    matrix<type>& current_matrix;
    unsigned number_rot, matrix_size;
    matrix<type> rotation_matrix;
    vector<type> eigenvalues;
    type precision;

    type NormOffDiagonal(){
        type Norm = 0;
        for(unsigned i = 0; i < matrix_size; ++i)
            for(unsigned j = 0; (j < matrix_size && i != j); ++j)
                Norm += current_matrix(i,j)*current_matrix(i,j);
        return Norm;
    }
    type NormDiagonal(){
        type Norm = 0;
        for(unsigned i = 0; i < matrix_size; ++i)
            Norm += current_matrix(i,i)*current_matrix(i,i);
        return Norm;
    }
    Jacoby_t(matrix<type>& input_matrix, type precision) : current_matrix(input_matrix), 
    precision(precision), matrix_size(input_matrix.size1()){
        type tetta, c, s;
        rotation_matrix = matrix<type>(matrix_size, matrix_size, 0.);
        matrix<type> rot;
        for(unsigned i = 0; i < matrix_size; ++i)
            rotation_matrix(i,i) = 1.0;
        number_rot = 0;
        eigenvalues.resize(matrix_size);
        while(NormOffDiagonal()/NormDiagonal() > precision){
            for(unsigned p = 0; p < matrix_size; ++p){
                for(unsigned q = p + 1; q < matrix_size; ++q){ 
                    tetta = atan2(2*current_matrix(p,q),current_matrix(q,q)-current_matrix(p,p))/2;
                    c = cos(tetta); s = sin(tetta);
                    rot = matrix<type>(matrix_size, matrix_size, 0.);
                    for(unsigned i = 0; i < matrix_size; ++i)
                        rot(i,i) = 1.0;
                    rot(p,p) = c; rot(q,q) = c;
                    rot(p,q) = s; rot(q,p) = -s;
                    current_matrix = prod(trans(rot),current_matrix);
                    current_matrix = prod(current_matrix, rot);
                    rotation_matrix = prod(rotation_matrix, rot);
                }
            }
            std::cout << NormOffDiagonal()/NormDiagonal() << std::endl;
            ++number_rot;
        }
        
        for(unsigned i=0; i < matrix_size; ++i)
            eigenvalues(i) = current_matrix(i,i);
    }

};

template<typename type, typename Type>
struct RungeKutta4{
    Type k1,k2,k3,k4;
    RungeKutta4(){}
    template<typename sysf>
    void do_step(sysf sysF, const Type &in, Type &out, type time, type step){
        sysF(in,k1,time);
        sysF(in+step*k1/2.,k2,time+step/2.);
        sysF(in+step*k2/2.,k3,time+step/2.);
        sysF(in+step*k3,k4,time+step);
        out=std::move(in + step*(k1+2.*k2+2.*k3+k4)/6.);
    }
};

template<typename type, typename Type>
struct LeapFrog{
    Type intermediate;
    bool first_in=1;
    LeapFrog(){}
    template<typename sysf>
    void do_step(sysf sysF, const Type &in, Type &out, type time, type step){
        if(first_in){
            sysF(in.first,intermediate.second,time);
            first_in=0;
        }
        intermediate.first=in.second+intermediate.second*step/2;
        out.first=in.first+intermediate.first*step;
        sysF(out.first,intermediate.second,time+step);
        out.second=intermediate.first+intermediate.second*step/2;
    }
};


template<typename type, typename Type>
struct Euler{
    Type k;
    Euler(){}
    template<typename sysf>
    void do_step(sysf sysF, const Type &in, Type &out, type time, type step){
        sysF(in,k,time);
        out=std::move(in + step*k);
    }
};


template<typename type, typename Type>
struct RungeKutta5_Fehlberg{
    Type k1,k2,k3,k4,k5,k6;
    RungeKutta5_Fehlberg(){}
    template<typename sysf>
    void do_step(sysf sysF, const Type &in, Type &out, type time, type step){
        sysF(in,k1,time);
        sysF(in+step*k1/4.,k2,time+step/4.);
        sysF(in+step*(3*k1/32.+9*k2/32.),k3,time+3*step/8);
        sysF(in+step*(1932*k1/2197-7200*k2/2197+7296*k3/2197),k4,time+12*step/13);
        sysF(in+step*(439*k1/216-8*k2+3680*k3/513-845*k4/4104),k5,time+step);
        sysF(in+step*(-8*k1/27+ 2*k2-3544*k3/2565+1859*k4/4104-11*k5/40),k6,time+step/2);
        
        out=std::move(in + step*(16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55));
    }
};

template<typename type, typename Integrator, typename sysf, typename observer,typename Type, 
typename controller = std::function<int(const Type& X, const Type& Y)>, 
typename exit=std::function<int(const Type&X, const type &t)> >
inline int integrate(Integrator rk, sysf sysF, Type& X0, type t, type h, int n, observer Obs,
                exit Bad = [](const Type& X, const type& t)->int{return 1;},
                controller Err = [](const Type& X, const Type& Y)->int{return -1;}){
    int state, number_steps = 0;
    Type Y;
    type h_begin=h;
    type T=t+h*n;
    Obs(X0,t);
    while(t<T && Bad(X0,t)){
        state=-1;
        std::cout << t << '\r';
        rk.do_step(sysF,X0,Y,t,h);
        if(Err(X0,Y)==-1){
            goto obs_point;
        } 
        while(Err(X0,Y)){
            h=h/2.;
            rk.do_step(sysF, X0,Y,t,h);
            //std::cout << "h_decrease= " << h << std::endl;
            //std::cout << "Y :" << std::endl << Y << std::endl;
            state=1;
        }
        if(state==1){
            //std::cout << "STOP1" << std::endl;
            h=h*2;
            //std::cout << "h_increase= " << h << std::endl;
            rk.do_step(sysF,X0,Y,t,h);
            goto obs_point;
        }
        while(!Err(X0,Y)){
            if(h > h_begin){
                state=0;
                break;
            }
            //std::cout << "STOP" << std::endl;
            h=h*2;
            //std::cout << "h_increase= " << h << std::endl;
            rk.do_step(sysF, X0,Y,t,h);
            state=0;
        }
        if(state==0){
            h=h/2;
            //std::cout << "h_decrease= " << h << std::endl; 
            rk.do_step(sysF,X0,Y,t,h);
        }
        obs_point:
        X0=Y;
        t+=h;
        ++number_steps;
        Obs(X0,t);
    }
    std::cout << std::endl;
    return number_steps;
}

state_vector cross_product(const state_vector &, const state_vector &);

template<typename point_t>
inline void reflection(const std::vector<point_t> &points, unsigned axis){
    std::vector<point_t> points_inter_x(points);
    std::for_each(points_inter_x.begin(), points_inter_x.end(), 
    [axis](point_t point){point.coord(axis) *= -1;});
    points.insert(points.end(), std::make_move_iterator(points_inter_x.begin()), 
    std::make_move_iterator(points_inter_x.end()));
}

template<typename type>
inline std::istream& operator>>(std::istream& is, std::vector<type>& in){
    for(auto& it : in)
        is >> it;
    return is;
}

template<typename T, typename U>
std::vector<T> operator/(const std::vector<T>& a, U b){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]/b);
    return c;
}
template<typename T, typename U>
std::vector<T> operator*(const std::vector<T>& a, U b){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]*b);
    return c;
}
template<typename T, typename U>
std::vector<T> operator*(U b, const std::vector<T>& a){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]*b);
    return c;
}

template<typename T>
std::vector<T> operator+(const std::vector<T>& a,const std::vector<T>& b){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]+b[i]);
    return c;
}

template<typename T>
std::vector<T> operator-(const std::vector<T>& a,const std::vector<T>& b){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]-b[i]);
    return c;
}


template<typename T, typename U>
std::vector<T> operator+(U b, const std::vector<T>& a){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]+b);
    return c;
}


template<typename T, typename U>
std::vector<T> operator+(const std::vector<T>& a,U b){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]+b);
    return c;
}

template<typename T, typename U>
std::vector<T> operator-(U b, const std::vector<T>& a){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]-b);
    return c;
}


template<typename T, typename U>
std::vector<T> operator-(const std::vector<T>& a,U b){
    std::vector<T> c(a.size());
    for(unsigned i=0;i<a.size();++i)
        c[i]=std::move(a[i]-b);
    return c;
}
