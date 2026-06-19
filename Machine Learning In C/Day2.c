#include <stdio.h>
float train[][2]={
    {0.0,0.0},
    {1.0,2.0},
    {2.0,4.0},
    {3.0,6.0},
    {4.0,8.0},
    {5.0,10.0},
    {6.0,12.0}
};
#define train_count (sizeof(train)/sizeof(train[0]))
float cost(float w,float b){
    float result=0.0;
    for (size_t i=0;i<train_count;i++){
        float x=train[i][0];
        float y=train[i][1];
        float y_predict=x*w;
        float d=y_predict-y;
        result+=d*d;
    }
    result /= train_count;
    return result;
}
int main(void){
    float w=0.0;
    float b=0.0;
    float eps=1e-3;
    float rate=1e-3;
    for (int i=0;i<1000;++i){
            float dw=(cost(w+eps,b)-cost(w,b))/eps;
            float db=(cost(w,b+eps)-cost(w,b))/eps;
            w-=rate*dw;
            b-=rate*db;
            printf("Cost : %f || w : %f || b : %f \n",cost(w,b),w,b);
    }

}