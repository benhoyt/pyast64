#include <stdio.h>

int main() {
    long long sum = 0;
    for (long long i = 0; i < 100000000; i++) {
        sum += i;
    }
    return (int)sum;
}
