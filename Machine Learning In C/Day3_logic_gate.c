#include <stdio.h>
#include <stdlib.h>
#include <math.h>
float sigmoid(float x){
    return 1.f / (1.f + expf(-x));
}
typedef float sample[3];
sample or_train[]={
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,1}
};
sample and_train[]={
    {0,0,0},
    {1,0,0},
    {0,1,0},
    {1,1,1}
};
sample nand_train[]={
    {0,0,1},
    {1,0,1},
    {0,1,1},
    {1,1,0}
};
sample *train=or_train;
size_t train_count=4;
float cost(float w1,float w2,float b){
    float result=0.0f;
    for (size_t i=0;i<train_count;++i){
        float x1=train[i][0];
        float x2=train[i][1];
        float y=sigmoid(x1*w1+x2*w2+b);
        float d=y-train[i][2];
        result+=d*d;
    }
    result /= train_count;
    return result;
}
int main(void){
    float w1=0.0;
    float w2=0.0;
    float b=0.0;
    float eps=1e-3;
    float rate=0.01;
    for (int i=0;i<10000;i++){
        float grad_w1=(cost(w1+eps,w2,b)-cost(w1,w2,b))/eps;
        float grad_w2=(cost(w1,w2+eps,b)-cost(w1,w2,b))/eps;
        float grad_b=(cost(w1,w2,b+eps)-cost(w1,w2,b))/eps;
        w1-=grad_w1*rate;
        w2-=grad_w2*rate;
        b-=grad_b*rate;
        printf("w1 : %f || w2 : %f || b : %f || cost : %f\n",w1,w2,b,cost(w1,w2,b));
    }
    for (int i=0;i<2;i++){
        for (int j=0;j<2;j++){
            printf("%d || %d || %f\n",i,j,sigmoid(w1*i+w2*j+b));
        }
    }
    return 0;
}