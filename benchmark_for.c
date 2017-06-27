#include <stdio.h>

int main() {
    int sum = 0;
    for (int i = 0; i < 100000000; i++) {
        sum += i;
    }
    return sum;
}
