---
title: C++程序设计复习笔记
published: 2025-05-24 17:35:09
tags: ["C++", "编程", "复习", "程序设计"] # 将 tags 修改为数组，并添加了相关标签
category: 学习笔记

---

<!--more-->

# C++程序设计复习笔记

## 第六章 指针、引用与动态内存分配

### 一、指针类型与指针变量

#### 1. 指针的基本概念

* **指针 (Pointer)**：一种复合数据类型，存储的是另一个变量的内存地址。
* 通过指针，可以间接访问和修改其指向的内存地址中的数据。
* **地址 (Address)**：内存中每个字节的唯一编号。

#### 2. 指针变量

* **指针变量的说明 (Declaration)**：

    ```cpp
    <数据类型> *<指针变量名>;
    // 例如:
    int *p; // p 是一个指向整型变量的指针
    char *name; // name 是一个指向字符型变量的指针
    ```

  * `*` 表明该变量是一个指针。
  * 数据类型指明了指针所指向的变量的类型。
  * 未初始化的指针处于**悬挂状态 (Dangling Pointer)**，其指向不确定，使用它可能导致程序崩溃。
* **指针变量的值 (Value)**：
  * 指针变量存储的是其所指向的内存单元的**首地址**。
  * 指针变量自身也占用内存空间（通常为4或8字节，取决于系统架构）。
* **指针可以指向的成分**：变量、常量、数组、函数、类对象等。

#### 3. 指针类型的初始化与赋值

* **初始化 (Initialization)**：
  * **初始化为内存地址**：使用取地址运算符 `&` 获取变量地址。

        ```cpp
        int a = 18;
        int *p = &a; // 指针p初始化为变量a的地址
        ```

  * **初始化为 `nullptr` (C++11及以后)**：表示指针不指向任何有效的内存地址。

        ```cpp
        int *p = nullptr;
        ```

  * **初始化为 `0` 或 `NULL` (C风格)**：与 `nullptr` 类似，表示空指针。

        ```cpp
        int *p1 = 0;
        int *p2 = NULL; // NULL通常是宏定义 (void*)0
        ```

* **赋值 (Assignment)**：
  * 赋值运算符 `=` 左边是指针变量，右边是地址表达式。
  * 地址存放数据的类型必须与指针说明的类型一致。
  * **不可任意赋一个内存地址常量**，只能赋已分配内存的变量地址或 `nullptr`。

        ```cpp
        int i = 3;
        int *pi;
        pi = &i; // 正确
        // pi = 1000; // 错误，不能直接赋地址常量
        ```

#### 4. 指针运算

* **取内容运算符 `*` (Dereference Operator)**：
  * 表达式：`*<指针变量名>`
  * 作用：访问指针所指向的内存地址中存储的数据。
  * `*p` 可以看作一个变量，可以出现在表达式的左边（赋值）或右边（取值）。

        ```cpp
        int a = 18, *p;
        p = &a;
        cout << *p; // 输出18 (a的值)
        *p = 20;    // 等价于 a = 20;
        ```

* **取地址运算符 `&` (Address-of Operator)**：
  * 表达式：`&<可寻址数据名>`
  * 作用：获取变量、数组元素、类对象等的内存地址。
* **指针的算术运算**：主要用于指向数组元素时。
  * `p + i`：指向指针 `p` 当前所指元素之后第 `i` 个元素的地址。
  * `p - i`：指向指针 `p` 当前所指元素之前第 `i` 个元素的地址。
  * `p++` 或 `++p`：指向下一个元素。
  * `p--` 或 `--p`：指向上一个元素。
  * `p2 - p1`：若 `p1` 和 `p2` 指向同一数组中的元素，结果是它们之间元素的个数。
  * **注意**：指针加减的单位是其所指向数据类型的大小。例如，`int *p; p+1;` 地址会增加 `sizeof(int)` 个字节。
* **指针的关系运算**：
  * 可以比较两个同类型指针是否相等 (`==`, `!=`) 或比较它们指向的地址先后 (`<`, `>`, `<=`, `>=`)。
  * 常用于判断指针是否为空 (`p == nullptr`) 或遍历数组。

### 二、指针与常量

#### 1. 指向常量的指针 (Pointer to Constant)

* 指针指向的内容不可通过该指针修改，但指针本身可以改变指向。
* 声明格式：`const <数据类型> *<指针变量名>;` 或 `<数据类型> const *<指针变量名>;`

    ```cpp
    const int a = 10;
    const int *p1 = &a; // p1 指向常量a
    // *p1 = 20; // 错误：不能通过p1修改a的值

    int b = 20;
    const int *p2 = &b; // p2 指向变量b，但视其内容为常量
    // *p2 = 30; // 错误：不能通过p2修改b的值
    b = 30;     // 正确：b本身可以修改
    int c = 40;
    p2 = &c;    // 正确：p2可以指向其他地址
    ```

* 常用于函数参数，防止函数内部修改实参。

#### 2. 常量指针 (Constant Pointer)

* 指针本身是常量，其指向的地址不可改变，但其指向地址的内容可以通过该指针修改（如果指向的不是常量）。
* 声明格式：`<数据类型> * const <指针变量名>;`
* **必须在声明时初始化。**

    ```cpp
    int a = 10;
    int b = 20;
    int * const p = &a; // p是一个常量指针，初始化后不能再指向其他地址
    *p = 15; // 正确：可以修改a的值
    // p = &b; // 错误：p不能指向其他地址
    ```

#### 3. 指向常量的常量指针 (Constant Pointer to Constant)

* 指针本身和其指向的内容都不可改变。
* 声明格式：`const <数据类型> * const <指针变量名>;`
* **必须在声明时初始化。**

    ```cpp
    const int a = 10;
    const int * const p = &a;
    // *p = 15; // 错误
    // int b = 20;
    // p = &b; // 错误
    ```

### 三、指针与数组

#### 1. 指向一维数组元素的指针

* **数组名即指针常量**：数组名代表数组首元素的地址。`a` 等价于 `&a[0]`。
* 可以用指针访问数组元素：

    ```cpp
    int arr[5] = {1, 2, 3, 4, 5};
    int *p = arr; // p 指向 arr[0]

    // 访问方式等价：
    // arr[i]  <==> *(arr + i) <==> *(p + i) <==> p[i]
    // &arr[i] <==> arr + i    <==> p + i
    ```

* 指针变量可以进行自增、自减等运算来移动指向，数组名作为常量指针不行。

    ```cpp
    p++; // p 指向 arr[1]
    // arr++; // 错误
    ```

#### 2. 指向二维数组元素的指针

* 二维数组 `A[m][n]` 在内存中是按行连续存储的。
* **`A`**：二维数组名，代表首行 `A[0]` 的地址，是一个指向包含 `n` 个元素的一维数组的指针。类型为 `(*)[n]`。
* **`A[i]`**：第 `i` 行的数组名，代表该行首元素 `A[i][0]` 的地址。类型为 `*`。
* **`&A[i][j]`**：第 `i` 行第 `j` 列元素的地址。
* **访问方式**：
  * `A[i][j]`
  * `*(A[i] + j)`
  * `*(*(A + i) + j)`
* **用指针访问二维数组元素**：
  * **行指针 (指向一维数组的指针)**：

        ```cpp
        int A[3][4];
        int (*p)[4]; // p是一个指针，指向包含4个int元素的一维数组
        p = A;       // p指向A[0]
        // p+i 指向 A[i]
        // *(p+i) 是 A[i] 的首元素地址，即 &A[i][0]
        // *(*(p+i)+j) 是 A[i][j] 的值
        // (*(p+i))[j] 也是 A[i][j] 的值
        ```

  * **元素指针 (指向单个元素的指针)**：

        ```cpp
        int A[3][4];
        int *p;
        p = A[0]; // p 指向 A[0][0] (或 p = &A[0][0];)
        // 此时p可以像一维数组指针一样遍历整个二维数组
        // p + (i * 列数 + j) 指向 A[i][j] 的地址
        ```

#### 3. 指针数组 (Array of Pointers)

* 数组的每个元素都是指针。
* 声明格式：`<数据类型> *<数组名>[<元素个数>];`

    ```cpp
    int a=1, b=2, c=3;
    int *ptrArr[3] = {&a, &b, &c}; // ptrArr是一个包含3个int*类型元素的数组
    // ptrArr[0] 指向 a, ptrArr[1] 指向 b, ptrArr[2] 指向 c
    ```

* 常用于存储多个字符串：

    ```cpp
    char *names[] = {"Alice", "Bob", "Charlie"};
    // names[0] 指向 "Alice" 的首字符 'A'
    ```

#### 4. 多重指针 (Pointer to Pointer)

* 指针变量本身也存储在内存中，也有地址。指向指针变量地址的指针称为多重指针。
* **二重指针**：指向一个指针变量的指针。

    ```cpp
    int x = 123;
    int *p = &x;    // p 指向 x
    int **q = &p;   // q 指向 p
    // *q  等价于 p (即x的地址)
    // **q 等价于 *p (即x的值，123)
    ```

* 常用于动态分配二维数组或处理指针数组。

#### 5. 字符指针与字符串

* 字符串可以用字符数组表示，也可以用字符指针指向字符串字面值常量。

    ```cpp
    char strArray[] = "hello"; // 字符数组，"hello"存储在数组中，可修改
    char *strPtr = "world";   // 字符指针，指向字符串常量"world"
                              // 字符串常量通常存储在只读区，不可通过strPtr修改
    ```

* `cout << strPtr;` 会输出整个字符串，直到遇到 `\0`。
* 字符指针数组常用于管理多个字符串。

    ```cpp
    char *days[] = {"Sunday", "Monday", ...};
    ```

* **字符串处理函数** (需包含 `<cstring>` 或 `<string.h>`)：
  * `strlen(const char *s)`：返回字符串长度（不包括 `\0`）。
  * `strcpy(char *dest, const char *src)`：复制字符串（有安全风险，建议用 `strcpy_s`）。
  * `strcat(char *dest, const char *src)`：连接字符串（有安全风险，建议用 `strcat_s`）。
  * `strcmp(const char *s1, const char *s2)`：比较字符串。

### 四、动态内存分配

#### 1. C++内存管理

* **静态存储区 (Static/Global Storage)**：存放全局变量、静态变量。程序运行期间一直存在。
* **栈区 (Stack)**：存放函数参数、局部变量。函数调用时分配，调用结束时自动释放。
* **堆区 (Heap)**：用于动态内存分配。程序员手动通过 `new` 分配，通过 `delete` 释放。

#### 2. 动态内存分配运算符

* **`new` 运算符**：在堆区分配内存。
  * 分配单个变量：`<指针变量> = new <数据类型>;`
  * 分配单个变量并初始化：`<指针变量> = new <数据类型>(<初值>);` 或 `<指针变量> = new <数据类型>{<初值>};` (C++11)
  * 分配一维数组：`<指针变量> = new <数据类型>[<元素个数>];`
  * 如果分配失败（内存不足），`new` 默认会抛出 `std::bad_alloc` 异常。
* **`delete` 运算符**：释放 `new` 分配的内存。
  * 释放单个变量：`delete <指针变量>;`
  * 释放一维数组：`delete [] <指针变量>;` (注意 `[]` 不可省略)
  * **重要**：
    * `delete` 只能用于 `new` 分配的内存。
    * 同一块内存不能 `delete` 多次。
    * `delete` 后，指针变量本身的值不会改变（仍指向原地址，成为悬挂指针），最好将其设为 `nullptr`。
    * `new` 和 `delete` (以及 `new[]` 和 `delete[]`)必须配对使用，否则会导致**内存泄漏 (Memory Leak)** 或 **未定义行为**。

#### 3. 动态变量

* 通过 `new <数据类型>` 创建的无名变量，通过指针访问。

    ```cpp
    int *p = new int(5); // 分配一个int空间，初始化为5
    cout << *p;          // 输出5
    delete p;            // 释放内存
    p = nullptr;
    ```

#### 4. 动态数组

* **一维动态数组**：

    ```cpp
    int n;
    cin >> n;
    int *arr = new int[n]; // 数组大小可以是变量
    for (int i = 0; i < n; ++i) {
        arr[i] = i * 10;
    }
    // ...使用arr...
    delete [] arr;
    arr = nullptr;
    ```

* **二维动态数组**：通常通过指针数组实现。

    ```cpp
    int rows, cols;
    cin >> rows >> cols;

    // 分配行指针数组
    int **matrix = new int*[rows];
    // 为每一行分配列空间
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new int[cols];
    }

    // 初始化和使用 matrix[i][j]
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = i * cols + j;
        }
    }

    // 释放内存 (与分配顺序相反)
    for (int i = 0; i < rows; ++i) {
        delete [] matrix[i]; // 先释放每一行
    }
    delete [] matrix; // 再释放行指针数组
    matrix = nullptr;
    ```

* **内存泄漏 (Memory Leak)**：如果动态分配的内存不再需要但没有被 `delete`，这块内存就无法再被程序使用，造成浪费。

### 五、引用类型

#### 1. 引用的含义

* **引用 (Reference)**：是已存在变量的一个**别名 (Alias)**。
* 引用不是新定义一个变量，它不占用独立的内存空间（或者说它与它所引用的变量共享同一块内存空间）。
* 对引用的操作就是对其所引用的变量的操作。

#### 2. 引用变量

* **引用变量的说明与初始化**：
  * 格式：`<数据类型> &<引用名> = <已存在变量名>;`
  * **引用在声明时必须初始化**，并且一旦初始化，就不能再引用其他变量。

        ```cpp
        int a = 10;
        int &ref_a = a; // ref_a 是 a 的引用 (别名)

        cout << a;      // 输出 10
        cout << ref_a;  // 输出 10

        ref_a = 20;     // 修改 ref_a 就是修改 a
        cout << a;      // 输出 20
        ```

  * `&` 在这里不是取地址运算符，而是引用声明符。

#### 3. 引用与指针的区别

| 特性         | 指针 (Pointer)                              | 引用 (Reference)                                 |
| :----------- | :------------------------------------------ | :----------------------------------------------- |
| **本质**     | 存储变量地址的变量                          | 已存在变量的别名                                 |
| **内存空间** | 占用独立的内存空间                          | 不占用独立内存空间 (与原变量共享)                |
| **初始化**   | 可以不初始化 (但危险)，可以为空 (`nullptr`) | 必须在声明时初始化，不能为空                     |
| **可变性**   | 可以改变指向 (指向不同变量)                 | 一旦初始化，不能再引用其他变量                   |
| **操作**     | 通过 `*` 间接访问，`p` 是地址，`*p` 是内容  | 直接使用，`ref` 就是内容                         |
| **多级**     | 可以有多级指针 (`int **p`)                  | 不能有引用的引用 (`int &&r` 非C++11右值引用含义) |
| **数组**     | 可以有指针数组 (`int *arr[]`)               | 不能有引用数组 (`int &arr[]` 非法)               |
| **空值**     | 可以为 `nullptr`                            | 必须引用一个存在的实体                           |

#### 4. 引用的安全性

* 引用比指针更安全，因为它必须初始化，且不能为空，也不能随意改变引用的对象。
* 常用于函数参数（传递大对象时避免拷贝开销，且能修改实参）和函数返回值。

    ```cpp
    // 通过引用修改参数
    void swap(int &x, int &y) {
        int temp = x;
        x = y;
        y = temp;
    }

    int main() {
        int m = 5, n = 10;
        swap(m, n); // m变为10, n变为5
        return 0;
    }
    ```

* **数组的引用**：

    ```cpp
    int arr[5];
    int (&ref_arr)[5] = arr; // ref_arr 是数组 arr 的引用
    ref_arr[0] = 10; // 等价于 arr[0] = 10;
    ```

## 第七章 类和对象

### 一、面向对象程序设计思想

#### 1. 结构化程序设计与面向对象程序设计

* **结构化程序设计**：以过程/函数为中心，自顶向下，逐步求精。
* **面向对象程序设计 (OOP)**：以对象为中心，将数据和操作数据的函数封装在一起。主要特征：封装、继承、多态。

#### 2. 类和对象的基本概念

* **类 (Class)**：
  * 对具有相同属性（数据）和行为（操作）的对象的抽象描述。
  * 一种用户自定义的数据类型，是创建对象的模板。
  * 定义类时不分配内存。
* **对象 (Object/Instance)**：
  * 类的具体实例。
  * 程序运行时，对象在内存中占据实际空间。
  * 每个对象都有其自身的状态（由数据成员的值决定）和行为（由成员函数实现）。

#### 3. 面向对象程序设计的特点

* **封装 (Encapsulation)**：将数据（属性）和操作这些数据的函数（方法）捆绑在一起，形成一个独立的单元（类）。对外部隐藏对象的内部实现细节，只提供公共接口进行交互。
* **继承 (Inheritance)**：允许创建一个新类（派生类/子类），从一个或多个已存在的类（基类/父类）继承属性和行为。实现代码重用和层次化组织。
* **多态 (Polymorphism)**：同一操作作用于不同的对象，可以有不同的解释，产生不同的执行结果。主要通过虚函数和重载实现。

#### 4. 面向对象程序设计的过程

1. **识别对象和类**：分析问题域，找出关键实体及其共同特征。
2. **定义类的属性和行为**：确定类的数据成员和成员函数。
3. **定义类之间的关系**：如继承、组合等。
4. **创建和使用对象**：通过类的实例解决问题。

### 二、类的定义与对象的说明

#### 1. 类的定义

* 使用 `class` 关键字。
* 一般格式：

    ```cpp
    class <类名> {
    private:
        // 私有成员 (数据和函数)
        // 只能被本类的成员函数和友元访问
    public:
        // 公有成员 (数据和函数)
        // 可以被类外部访问，是类的接口
    protected:
        // 保护成员 (数据和函数)
        // 可以被本类的成员函数、友元以及派生类的成员函数访问
    // friend <友元声明>; // 友元不是成员，但可以访问私有和保护成员
    }; // 注意类定义末尾的分号
    ```

* **访问限定符 (Access Specifiers)**：`private`, `public`, `protected`。
  * 若不写访问限定符，类中成员默认为 `private` (结构体 `struct` 中默认为 `public`)。
  * 访问限定符可以出现多次，其作用域到下一个访问限定符或类定义结束。

#### 2. 成员变量与成员函数

* **成员变量 (Data Members / Attributes)**：描述类的属性，可以是任何数据类型（包括其他类的对象）。
  * 可以在声明时赋默认值 (C++11)。
* **成员函数 (Member Functions / Methods)**：描述类的行为，操作类的数据成员。
  * **类内定义**：函数体直接写在类定义中，通常自动成为内联函数 (inline)。
  * **类外定义**：在类定义中只写函数原型，在类外部使用作用域解析运算符 `::` 定义函数体。

        ```cpp
        class MyClass {
        public:
            void func1(); // 类内声明
            void func2() { /* 类内定义 */ }
        };

        void MyClass::func1() { // 类外定义
            // ...
        }
        ```

#### 3. 类对象的说明 (创建对象)

* **普通对象**：`<类名> <对象名>;` 或 `<类名> <对象名>(<构造函数实参>);`
* **对象数组**：`<类名> <数组名>[<大小>];`
* **对象指针**：`<类名> *<指针名>;`
  * 通常与 `new` 结合使用动态创建对象：`指针名 = new <类名>(<构造函数实参>);`
  * 访问成员：`指针名->成员` 或 `(*指针名).成员`
* **对象引用**：`<类名> &<引用名> = <已存在对象>;`

#### 4. `this` 指针

* 是一个隐含的指针，存在于类的非静态成员函数中。
* 指向调用该成员函数的**当前对象**。
* 值为当前对象的起始地址。
* 主要用途：
  * 区分同名的成员变量和局部变量/参数：`this->成员变量`。
  * 在成员函数中返回当前对象的引用或指针：`return *this;` 或 `return this;`。

    ```cpp
    class Box {
    public:
        int H;
        void setH(int H) {
            this->H = H; // this->H 是成员变量, H 是参数
        }
    };
    ```

### 三、构造函数与析构函数

#### 1. 对象的初始化

* 对象在使用前通常需要初始化其数据成员。
* 可以通过构造函数自动完成。

#### 2. 构造函数 (Constructor)

* 一种特殊的成员函数，在创建对象时自动被调用，用于初始化对象。
* **特点**：
  * 函数名与类名完全相同。
  * 没有返回类型 (连 `void` 也没有)。
  * 可以重载 (定义多个参数列表不同的构造函数)。
  * 可以有参数，可以有默认参数。
* **默认构造函数 (Default Constructor)**：
  * 无参数的构造函数。
  * 如果类中没有定义任何构造函数，编译器会自动生成一个不做任何事情的默认构造函数。
  * 一旦定义了任何构造函数，编译器就不再自动生成默认构造函数。如果此时仍需要无参构造，必须显式定义或使用 `= default` (C++11)。

        ```cpp
        class MyClass {
        public:
            MyClass() = default; // 显式声明使用编译器生成的默认构造函数
            // MyClass() {}      // 自定义默认构造函数
        };
        ```

* **自定义构造函数**：
  * **初始化列表 (Member Initializer List)**：推荐使用，在构造函数体执行前初始化成员。效率更高，特别是对于类类型成员和 `const` 或引用成员（它们必须在初始化列表初始化）。

        ```cpp
        class Point {
            int x, y;
        public:
            Point(int ix, int iy) : x(ix), y(iy) { // 初始化列表
                // 构造函数体
            }
        };
        ```

* **委托构造函数 (Delegating Constructor) (C++11)**：一个构造函数可以在其初始化列表中调用同一个类的另一个构造函数。

    ```cpp
    class Box {
        double l, w, h;
    public:
        Box(double s) : Box(s, s, s) {} // 委托给三参数构造函数
        Box(double lv, double wv, double hv) : l(lv), w(wv), h(hv) {}
    };
    ```

* **`explicit` 关键字**：用于修饰单参数构造函数（或所有参数都有默认值的多参数构造函数），防止隐式类型转换。

    ```cpp
    class String {
    public:
        explicit String(int size); // 防止 int 隐式转换为 String
        String(const char *s);
    };
    // String s1 = 10; // 错误，如果构造函数是 explicit
    String s2(10); // 正确
    ```

#### 3. 析构函数 (Destructor)

* 一种特殊的成员函数，在对象生命周期结束时（如对象离开作用域、`delete` 指向对象的指针时）自动被调用。
* 主要用于释放对象占用的资源（如动态分配的内存、打开的文件等）。
* **特点**：
  * 函数名是 `~` 后跟类名 (例如 `~MyClass()`)。
  * 没有返回类型，没有参数。
  * 一个类只能有一个析构函数，不能重载。
  * 如果类中没有定义析构函数，编译器会自动生成一个不做任何事情的默认析构函数。
  * 如果类中管理了动态资源（如构造函数中 `new` 了内存），通常需要自定义析构函数来 `delete` 这些资源。

    ```cpp
    class MyArray {
        int *data;
    public:
        MyArray(int size) { data = new int[size]; }
        ~MyArray() {
            delete [] data; // 释放动态分配的内存
            data = nullptr;
        }
    };
    ```

* **构造和析构顺序**：
  * 局部对象：后构造的先析构（栈特性）。
  * 数组成员/基类成员：按声明/继承顺序构造，按相反顺序析构。

#### 4. 拷贝构造函数 (Copy Constructor)

* 一种特殊的构造函数，用于使用一个已存在的同类对象来初始化一个新创建的对象。
* **调用时机**：
    1. 用一个对象初始化另一个对象：`MyClass obj2 = obj1;` 或 `MyClass obj2(obj1);`
    2. 对象作为函数参数按值传递时。
    3. 函数返回值为对象时。
* **原型**：`<类名>(const <类名> &<引用名>);` (通常参数是 `const` 引用)
* **默认拷贝构造函数**：如果类没有显式定义拷贝构造函数，编译器会自动生成一个，它执行**浅拷贝 (Shallow Copy)**，即逐个复制非静态成员的值。
* **深拷贝 (Deep Copy)**：
  * 当类中包含指针成员，并且这些指针指向动态分配的资源时，浅拷贝会导致问题（多个对象指向同一块内存，一个对象析构时释放内存，其他对象指针悬挂；重复释放）。
  * 此时需要自定义拷贝构造函数，为新对象重新分配独立的资源，并复制内容，这就是深拷贝。

    ```cpp
    class MyString {
        char *str;
    public:
        MyString(const char *s = "") {
            str = new char[strlen(s) + 1];
            strcpy(str, s);
        }
        // 深拷贝构造函数
        MyString(const MyString &other) {
            str = new char[strlen(other.str) + 1];
            strcpy(str, other.str);
        }
        ~MyString() { delete [] str; }
        // ... 可能还需要重载赋值运算符=
    };
    ```

* **赋值运算符重载 (`operator=`)**：与拷贝构造函数类似，当类管理动态资源时，通常也需要重载赋值运算符以实现深拷贝赋值，防止自我赋值，并正确处理原有资源。

### 四、常对象与常量成员

#### 1. 常对象 (Constant Object)

* 用 `const` 修饰的对象。
* 声明格式：`const <类名> <对象名>(<实参表>);` 或 `<类名> const <对象名>(<实参表>);`
* 常对象的数据成员在对象创建后不能被修改。
* **常对象只能调用常量成员函数和访问公有常量数据成员。**

#### 2. 常量成员

* **常量数据成员 (Constant Data Member)**：
  * 用 `const` 修饰的数据成员。
  * **必须在构造函数的初始化列表中进行初始化。**
  * 一旦初始化，其值不能再改变。

        ```cpp
        class Test {
            const int MAX_SIZE;
        public:
            Test(int size) : MAX_SIZE(size) {} // 必须在初始化列表
        };
        ```

* **常量成员函数 (Constant Member Function)**：
  * 在函数声明和定义的参数列表后加 `const` 关键字。
  * 格式：`<返回类型> <函数名>(<参数表>) const;`
  * 常量成员函数**不能修改对象的数据成员** (除非数据成员被 `mutable` 修饰)。
  * 常量成员函数可以被普通对象和常对象调用。
  * 普通成员函数不能被常对象调用。
  * 在常量成员函数中，`this` 指针是一个指向常对象的指针 (`const <类名>* const this`)。

        ```cpp
        class Point {
            int x, y;
        public:
            Point(int x, int y) : x(x), y(y) {}
            int getX() const { return x; } // 常量成员函数
            // void setX(int val) { x = val; } // 普通成员函数
            void setX(int val) const { /* x = val; 错误 */ } // 尝试修改会报错
        };
        // const Point p1(1,2);
        // p1.getX(); // 正确
        // p1.setX(3); // 错误，如果setX不是const
        ```

* `mutable` 关键字：用于修饰数据成员，表示该成员即使在 `const` 成员函数或 `const` 对象中也可以被修改。

### 五、静态成员 (Static Members)

#### 1. 静态数据成员 (Static Data Members)

* 用 `static` 关键字修饰的数据成员。
* **属于类本身，而不是类的某个特定对象。** 该类的所有对象共享同一个静态数据成员的副本。
* **必须在类定义之外进行初始化和定义** (除非是 `const static` 整型或枚举型，可以在类内初始化 C++11/17)。

    ```cpp
    class Account {
    public:
        static double interestRate; // 声明静态数据成员
        // ...
    };
    double Account::interestRate = 0.025; // 定义并初始化静态数据成员
    ```

* **访问方式**：
  * 通过类名和作用域解析运算符：`<类名>::<静态数据成员名>`
  * 通过对象名：`<对象名>.<静态数据成员名>`
  * 通过对象指针：`<指针名>-><静态数据成员名>`
* 主要用途：统计对象个数、共享数据等。

#### 2. 静态成员函数 (Static Member Functions)

* 用 `static` 关键字修饰的成员函数。
* **不依赖于任何特定对象，没有 `this` 指针。**
* **只能直接访问类的静态数据成员和调用其他静态成员函数。** 不能直接访问非静态成员（除非通过传递对象引用或指针）。
* **访问方式**：
  * 通过类名和作用域解析运算符：`<类名>::<静态成员函数名>()`
  * 通过对象名：`<对象名>.<静态成员函数名>()`
  * 通过对象指针：`<指针名>-><静态成员函数名>()`

    ```cpp
    class Counter {
        static int count;
    public:
        Counter() { count++; }
        static int getCount() { return count; } // 静态成员函数
    };
    int Counter::count = 0; // 初始化

    // Counter c1, c2;
    // cout << Counter::getCount(); // 通过类名调用
    ```

### 六、友元 (Friends)

#### 1. 友元的基本概念

* 友元是一种机制，允许一个类授予其他类或函数访问其 `private` 和 `protected` 成员的权限。
* 友元声明在类定义内部，使用 `friend` 关键字。
* 友元关系是单向的、不传递的。

#### 2. 友元函数 (Friend Functions)

* 一个普通函数或另一个类的成员函数可以被声明为一个类的友元。
* 友元函数不是类的成员函数，但可以访问该类的所有成员。
* **普通函数作为友元**：

    ```cpp
    class MyClass {
        int data;
    public:
        MyClass(int d) : data(d) {}
        friend void showData(const MyClass &obj); // 友元函数声明
    };

    void showData(const MyClass &obj) {
        cout << obj.data << endl; // 可以访问私有成员 data
    }
    ```

* **其他类的成员函数作为友元**：

    ```cpp
    class B; // 前向声明
    class A {
        int val_a;
    public:
        A(int v) : val_a(v) {}
        friend void B::displayA(const A &objA);
    };

    class B {
    public:
        void displayA(const A &objA) {
            cout << objA.val_a << endl; // B的成员函数访问A的私有成员
        }
    };
    ```

#### 3. 友元类 (Friend Classes)

* 一个类B可以被声明为另一个类A的友元类。这意味着类B的所有成员函数都是类A的友元函数，可以访问类A的所有成员。

    ```cpp
    class A {
        int secret;
    public:
        A(int s) : secret(s) {}
        friend class B; // B是A的友元类
    };

    class B {
    public:
        void revealSecret(const A &objA) {
            cout << objA.secret << endl; // B可以访问A的私有成员
        }
    };
    ```

### 七、类与类之间的关系

#### 1. 类的对象成员 (Composition / Aggregation)

* 一个类的数据成员可以是另一个类的对象。这表示 "has-a" 关系。
* **构造顺序**：当创建包含对象成员的类的对象时，会先调用对象成员的构造函数（按照在类中声明的顺序），然后再执行宿主类的构造函数体。
* **初始化**：对象成员的初始化通常在宿主类的构造函数的初始化列表中进行。

    ```cpp
    class Engine { /* ... */ };
    class Car {
        Engine carEngine; // Engine对象是Car的成员
    public:
        Car(...) : carEngine(...) { /* ... */ }
    };
    ```

#### 2. 类的嵌套 (Nested Classes)

* 一个类可以定义在另一个类的内部。
* 嵌套类（内部类）的作用域在外围类之内。
* 访问权限规则依然适用。

    ```cpp
    class Outer {
    public:
        class Inner { // 嵌套类
        public:
            void display() { /* ... */ }
        };
        void useInner() {
            Inner in;
            in.display();
        }
    };
    // Outer::Inner obj; // 如果Inner是public的
    ```

### 八、类中的运算符重载

#### 1. 运算符重载概述

* 允许为自定义类类型的对象重新定义C++中已有运算符的含义。
* 使得对类对象的操作更自然、直观，类似基本数据类型。
* 通过定义特殊的成员函数或友元函数（称为运算符函数）来实现。
* 函数名格式：`operator<运算符符号>` (例如 `operator+`, `operator==`)。
* **限制**：
  * 不能创建新的运算符。
  * 不能改变运算符的优先级、结合性或操作数个数。
  * `.`, `::`, `?:`, `sizeof`, `.*` 这五个运算符不能被重载。
  * `=`, `()`, `[]`, `->` 必须作为类的成员函数重载。

#### 2. 友元方式重载运算符

* 运算符函数是类的友元。
* 操作数都作为函数的显式参数。
* 对于二元运算符 `op`，`obj1 op obj2` 相当于 `operator op(obj1, obj2)`。
* 对于一元运算符 `op`，`op obj` 相当于 `operator op(obj)`。

    ```cpp
    class Complex {
        double real, imag;
    public:
        Complex(double r=0, double i=0) : real(r), imag(i) {}
        friend Complex operator+(const Complex &c1, const Complex &c2);
        void display() const { cout << real << " + " << imag << "i" << endl; }
    };
    Complex operator+(const Complex &c1, const Complex &c2) {
        return Complex(c1.real + c2.real, c1.imag + c2.imag);
    }
    // Complex c1(1,2), c2(3,4); Complex c3 = c1 + c2;
    ```

#### 3. 成员方式重载运算符

* 运算符函数是类的成员函数。
* 左操作数是调用该函数的对象 (`this` 指针指向它)。
* 对于二元运算符 `op`，`obj1 op obj2` 相当于 `obj1.operator op(obj2)`。右操作数是参数。
* 对于一元运算符 `op`，`op obj` 相当于 `obj.operator op()`。无参数。

    ```cpp
    class Complex {
        double real, imag;
    public:
        Complex(double r=0, double i=0) : real(r), imag(i) {}
        Complex operator+(const Complex &other) const { // 成员函数
            return Complex(this->real + other.real, this->imag + other.imag);
        }
        void display() const { cout << real << " + " << imag << "i" << endl; }
    };
    // Complex c1(1,2), c2(3,4); Complex c3 = c1 + c2; // c1.operator+(c2)
    ```

* **输入/输出运算符 (`<<`, `>>`) 重载**：通常作为友元函数重载，因为左操作数是流对象 (`ostream` 或 `istream`)。

    ```cpp
    class Point {
        int x, y;
    public:
        Point(int x=0, int y=0) : x(x), y(y) {}
        friend ostream& operator<<(ostream &out, const Point &p);
        friend istream& operator>>(istream &in, Point &p);
    };
    ostream& operator<<(ostream &out, const Point &p) {
        out << "(" << p.x << ", " << p.y << ")";
        return out;
    }
    istream& operator>>(istream &in, Point &p) {
        in >> p.x >> p.y;
        return in;
    }
    ```

### 九、简单的数据结构设计 (使用类)

* **链表 (Linked List)**：
  * 节点类 (Node)：包含数据域和指向下一个节点的指针域。
  * 链表类 (List)：包含头指针 (head)、尾指针 (tail)，以及插入、删除、查找、打印等操作。
* **栈 (Stack)**：后进先出 (LIFO)。
  * 可以用数组或链表实现。
  * 操作：`push` (入栈), `pop` (出栈), `top` (查看栈顶元素), `isEmpty`。
* **队列 (Queue)**：先进先出 (FIFO)。
  * 可以用数组或链表实现。
  * 操作：`enqueue` (入队), `dequeue` (出队), `front` (查看队头元素), `isEmpty`。

## 第八章 类的继承与多态性

### 一、类的继承与派生

#### 1. 继承与派生的基本概念

* **继承 (Inheritance)**：一种机制，允许一个类（派生类）获取另一个类（基类）的属性和方法。
* **派生 (Derivation)**：从基类创建派生类的过程。
* **基类 (Base Class / Parent Class / Superclass)**：被继承的类。
* **派生类 (Derived Class / Child Class / Subclass)**：通过继承创建的新类。
* **目的**：代码重用、建立类之间的层次关系 ("is-a" 关系)。

#### 2. 继承的种类

* **单继承 (Single Inheritance)**：一个派生类只有一个直接基类。
* **多级继承 (Multilevel Inheritance)**：派生类本身又可以作为另一个类的基类，形成继承链 (e.g., A -> B -> C)。
* **多重继承 (Multiple Inheritance)**：一个派生类可以有多个直接基类。

### 二、派生类

#### 1. 派生类的定义

* 语法：

    ```cpp
    class <派生类名> : <继承方式> <基类名1>, <继承方式> <基类名2>, ... {
        // 派生类新增的成员
    };
    ```

* **继承方式 (Access Specifier for Inheritance)**：`public`, `protected`, `private`。
  * **`public` 继承** (最常用)：
    * 基类的 `public` 成员在派生类中仍为 `public`。
    * 基类的 `protected` 成员在派生类中仍为 `protected`。
    * 基类的 `private` 成员在派生类中不可直接访问 (但被继承了)。
  * **`protected` 继承**：
    * 基类的 `public` 和 `protected` 成员在派生类中都变为 `protected`。
    * 基类的 `private` 成员在派生类中不可直接访问。
  * **`private` 继承**：
    * 基类的 `public` 和 `protected` 成员在派生类中都变为 `private`。
    * 基类的 `private` 成员在派生类中不可直接访问。
  * 如果省略继承方式，默认为 `private` (如果用 `class` 定义派生类) 或 `public` (如果用 `struct` 定义派生类)。

#### 2. 派生类的构造函数与析构函数

* **构造函数**：
  * 派生类不能继承基类的构造函数。
  * 派生类构造函数必须负责初始化其基类部分的成员和派生类新增的成员。
  * **调用基类构造函数**：在派生类构造函数的**初始化列表**中显式调用基类的构造函数。

        ```cpp
        class Base {
        public:
            Base(int b_val) : b_data(b_val) { cout << "Base constructor" << endl; }
        private:
            int b_data;
        };

        class Derived : public Base {
        public:
            Derived(int b_val, int d_val) : Base(b_val), d_data(d_val) { // 调用基类构造函数
                cout << "Derived constructor" << endl;
            }
        private:
            int d_data;
        };
        // Derived d(10, 20);
        ```

  * **构造顺序**：
        1. (虚)基类构造函数 (按继承声明顺序，若有虚继承则特殊处理)。
        2. 直接基类构造函数 (按继承声明顺序)。
        3. 类成员对象构造函数 (按在类中声明顺序)。
        4. 派生类构造函数体。
  * 如果派生类构造函数初始化列表中没有显式调用基类构造函数，编译器会尝试调用基类的默认（无参）构造函数。如果基类没有默认构造函数，则编译错误。
* **析构函数**：
  * 派生类不能继承基类的析构函数，但会自动调用。
  * **析构顺序**与构造顺序相反：
        1. 派生类析构函数体。
        2. 类成员对象析构函数 (按声明逆序)。
        3. 直接基类析构函数 (按继承声明逆序)。
        4. (虚)基类析构函数。
  * **重要**：如果基类指针可能指向派生类对象，并且需要通过基类指针 `delete` 对象，则基类的析构函数**必须声明为虚析构函数 (`virtual ~Base()`)**，以确保正确调用派生类的析构函数，防止资源泄漏。

#### 3. 友元与静态成员的继承

* **友元关系不能被继承**。基类的友元不是派生类的友元，派生类的友元也不是基类的友元。
* **静态成员被继承**。基类的静态成员（如果可访问）成为派生类的一部分，并且仍然是静态的，被所有基类和派生类对象共享。

#### 4. 派生类与基类的赋值兼容性 (Type Compatibility)

* **公有继承 (`public`) 体现 "is-a" 关系**，使得派生类对象可以被当作基类对象使用。
* **赋值兼容规则**：
    1. **派生类对象可以赋值给基类对象（对象切片 Object Slicing）**：

        ```cpp
        Base b_obj;
        Derived d_obj;
        b_obj = d_obj; // 合法，但只复制基类部分，派生类特有成员丢失
        ```

    2. **派生类对象的地址可以赋值给基类指针**：

        ```cpp
        Base *b_ptr;
        Derived d_obj;
        b_ptr = &d_obj; // 合法，b_ptr 指向 d_obj 的基类部分
        // b_ptr->derived_member; // 错误，只能访问基类成员
        ```

    3. **派生类对象可以初始化基类的引用**：

        ```cpp
        Derived d_obj;
        Base &b_ref = d_obj; // 合法，b_ref 引用 d_obj 的基类部分
        ```

* 反向操作（基类对象赋值给派生类对象、基类指针指向派生类等）通常不被允许或不安全，除非进行显式类型转换且确保类型正确。

#### 5. 同名隐藏 (Name Hiding / Overriding Data Members)

* 如果派生类定义了一个与基类成员（数据成员或成员函数）同名的成员，则派生类中的成员会**隐藏 (hide)** 基类中的同名成员。
* 在派生类作用域内，直接使用该名称访问的是派生类的成员。
* 要访问被隐藏的基类成员，需要使用作用域解析运算符：`<基类名>::<成员名>`。

    ```cpp
    class Base { public: int x = 1; void print() { cout << "Base x: " << x << endl; } };
    class Derived : public Base { public: int x = 2; void print() { cout << "Derived x: " << x << endl; } };
    // Derived d;
    // d.x; // 访问 Derived::x (值为2)
    // d.Base::x; // 访问 Base::x (值为1)
    // d.print(); // 调用 Derived::print()
    // d.Base::print(); // 调用 Base::print()
    ```

### 三、虚基类与虚拟继承

#### 1. 二义性问题 (Ambiguity)

* **多重继承中的二义性**：如果一个派生类从多个基类继承了同名成员，直接访问该成员会导致二义性。需要用类名限定。
* **菱形继承 (Diamond Problem)**：

    ```
        A
       / \
      B   C
       \ /
        D
    ```

    如果类 B 和类 C 都继承自类 A，然后类 D 同时继承自类 B 和类 C，那么类 D 的对象中会包含**两份**类 A 的成员副本（一份来自 B，一份来自 C）。这会导致访问 A 的成员时产生二义性，并且浪费内存。

#### 2. 虚基类与虚拟继承 (Virtual Base Class & Virtual Inheritance)

* 用于解决菱形继承问题。
* 当一个基类被声明为虚基类时，在后续的派生类中，无论该虚基类通过多少条路径被间接继承，派生类对象中都只包含该虚基类的一个共享副本。
* **声明虚基类**：在派生类继承基类时使用 `virtual` 关键字。

    ```cpp
    class A { /* ... */ };
    class B : virtual public A { /* ... */ }; // B虚拟继承A
    class C : virtual public A { /* ... */ }; // C虚拟继承A
    class D : public B, public C { /* ... */ }; // D中只有一份A的副本
    ```

* **虚基类的构造**：虚基类的构造函数由**最终派生类 (Most Derived Class)** 的构造函数负责调用。中间基类的构造函数初始化列表中对虚基类的调用会被忽略（除非该中间基类本身就是最终派生类）。

### 四、多态性与虚函数

#### 1. 多态性 (Polymorphism)

* "多种形态"。指同样的消息（函数调用）被不同类型的对象接收时导致不同的行为。
* **静态多态 (Compile-time Polymorphism / Early Binding)**：
  * 通过函数重载 (Function Overloading) 和运算符重载 (Operator Overloading) 实现。
  * 在编译时确定调用哪个函数。
* **动态多态 (Run-time Polymorphism / Late Binding)**：
  * 通过虚函数 (Virtual Functions) 和继承实现。
  * 在运行时根据对象的实际类型确定调用哪个函数。
  * 是面向对象编程的核心特性之一。

#### 2. 虚函数 (Virtual Function)

* 在基类中使用 `virtual` 关键字声明的成员函数。
* 当派生类重写 (override) 基类的虚函数时（函数名、参数列表、返回类型、`const`属性都相同），通过基类指针或引用调用该虚函数，会根据指针或引用实际指向的对象的类型来调用相应的版本。

    ```cpp
    class Shape {
    public:
        virtual void draw() { cout << "Drawing a generic shape." << endl; }
        virtual ~Shape() {} // 虚析构函数
    };
    class Circle : public Shape {
    public:
        void draw() override { cout << "Drawing a circle." << endl; } // C++11 override 关键字
    };
    class Square : public Shape {
    public:
        void draw() override { cout << "Drawing a square." << endl; }
    };

    // void showDrawing(Shape *s) { s->draw(); }
    // Circle c; Square sq;
    // showDrawing(&c);  // 输出 "Drawing a circle."
    // showDrawing(&sq); // 输出 "Drawing a square."
    ```

* **`override` 关键字 (C++11)**：在派生类中重写虚函数时使用，编译器会检查基类是否存在对应的可重写的虚函数，有助于防止错误。
* **`final` 关键字 (C++11)**：
  * 用于虚函数：表示该虚函数不能在更深层次的派生类中被重写。
  * 用于类：表示该类不能被继承。
* 构造函数不能是虚函数。
* 析构函数通常应声明为虚函数，特别是当基类指针可能用于删除派生类对象时。

#### 3. 纯虚函数与抽象基类 (Pure Virtual Function & Abstract Base Class)

* **纯虚函数**：在基类中声明但没有定义的虚函数，其声明末尾加 `= 0;`。

    ```cpp
    class AbstractShape {
    public:
        virtual void calculateArea() = 0; // 纯虚函数
        virtual ~AbstractShape() {}
    };
    ```

* **抽象基类 (Abstract Base Class, ABC)**：包含至少一个纯虚函数的类。
  * **不能创建抽象基类的对象实例。**
  * 主要用作接口，强制派生类实现纯虚函数。
  * 如果派生类没有实现基类中的所有纯虚函数，那么该派生类仍然是抽象基类。
  * 可以声明抽象基类的指针或引用，指向其具体派生类的对象。

## 第九章 类模板与STL程序设计

### 一、函数模板

#### 1. 函数模板的基本概念

* **函数模板 (Function Template)**：一个通用的函数描述，其中的数据类型使用类型参数（模板参数）表示。
* 编译器根据函数调用时提供的实参类型，自动推断模板参数的具体类型，并生成一个特定版本的函数实例（模板实例化）。
* 定义格式：

    ```cpp
    template <typename <类型参数名1>, typename <类型参数名2>, ...> // 或 class 代替 typename
    <返回类型> <函数名>(<参数列表>) {
        // 函数体，可以使用类型参数名
    }
    ```cpp
    template <typename T>
    T maxVal(T a, T b) {
        return (a > b) ? a : b;
    }
    // int m = maxVal(10, 20);       // T 被推断为 int
    // double d = maxVal(3.14, 2.71); // T 被推断为 double
    ```

* 函数模板调用时不进行实参到形参类型的自动转换（除非显式指定模板参数）。

#### 2. 函数模板的特例 (Specialization)

* 为特定的数据类型提供一个不同于通用模板的特殊实现。
* 格式：`template <> <返回类型> <函数名><<特化类型>>(<参数列表>) { ... }`

    ```cpp
    // 通用模板
    template <typename T>
    int compare(T a, T b) { return a > b ? 1 : (a < b ? -1 : 0); }

    // 针对 const char* 的特化版本
    template <>
    int compare<const char*>(const char* a, const char* b) {
        return strcmp(a, b);
    }
    ```

#### 3. 函数模板的重载 (Overloading)

* 可以定义多个同名函数模板，只要它们的模板参数列表或函数参数列表不同。
* 也可以定义同名的普通函数。调用时，编译器优先匹配普通函数，其次是更特化的模板，最后是通用模板。

### 二、类模板的基本概念

#### 1. 类模板的定义

* **类模板 (Class Template)**：一个通用的类描述，允许类的数据成员、成员函数的参数或返回类型使用类型参数。
* 定义格式：

    ```cpp
    template <typename <类型参数名1>, int <非类型参数名1>, ...> // 可以有类型参数和非类型参数
    class <类模板名> {
        // 类成员，可以使用模板参数
    };
    ```

  * **类型参数 (Type Parameter)**：用 `typename` 或 `class` 声明，代表一种数据类型。
  * **非类型参数 (Non-type Parameter)**：如 `int N`，代表一个常量值，其实参必须是常量表达式。

#### 2. 类模板的实例化

* 使用类模板创建对象时，必须显式指定所有模板参数的具体类型或值。
* 格式：`<类模板名><<实参列表>> <对象名>;`

    ```cpp
    template <typename T, int SIZE>
    class Array {
        T data[SIZE];
    public:
        // ...
    };
    // Array<int, 10> intArray;   // T 为 int, SIZE 为 10
    // Array<double, 5> doubleArray; // T 为 double, SIZE 为 5
    ```

#### 3. 类模板的成员函数

* 可以在类模板定义内定义，也可以在类模板定义外定义。
* **类外定义格式**：

    ```cpp
    template <typename T, int SIZE>
    void Array<T, SIZE>::someFunction() {
        // ...
    }
    ```

    每一处使用类模板名的地方都需要带上模板参数列表 `<T, SIZE>`。

#### 4. 类模板的静态成员与友元

* **静态成员**：每个实例化后的类模板都有其自己的一份静态成员。
  * 静态数据成员的定义和初始化仍在类外，但需要 `template <...>` 前缀和类名后的 `<...>`。

        ```cpp
        template <typename T>
        class Counter {
        public:
            static int count;
            Counter() { count++; }
        };
        template <typename T> // 不能少
        int Counter<T>::count = 0; // 初始化
        // Counter<int> c1; Counter<double> c2; // Counter<int>::count 和 Counter<double>::count 是不同的
        ```

* **友元**：
  * 普通函数/类作友元：是所有实例化类的友元。
  * 函数模板/类模板作友元：情况较复杂，可以指定特定实例化为友元。

#### 5. 类模板的特例版本 (Specialization)

* 为特定的模板参数组合提供一个完全不同的类定义。

    ```cpp
    // 通用类模板
    template <typename T> class Storage { T data; /* ... */ };

    // 针对 bool 的特化版本
    template <> class Storage<bool> { unsigned char bits; /* ... 特殊实现 ... */ };
    ```

* 也可以只特化部分模板参数（偏特化，Partial Specialization）。

### 三、类模板的继承和派生

* 类模板可以参与继承关系。
    1. **普通类派生类模板**：基类是普通类，派生类是类模板。
    2. **类模板派生普通类**：基类是类模板的特定实例化，派生类是普通类。

        ```cpp
        template <typename T> class BaseT { T data; };
        class Derived : public BaseT<int> { /* ... */ }; // 派生自 BaseT<int>
        ```

    3. **类模板派生类模板**：

        ```cpp
        template <typename T> class BaseT { /* ... */ };
        template <typename U> class DerivedT : public BaseT<U> { /* ... */ };
        template <typename T1, typename T2> class DerivedMultiT : public BaseT<T1> { T2 other_data; };
        ```

### 四、标准模板库 (STL) 程序设计

#### 1. STL基本概念

* **STL (Standard Template Library)**：C++标准库的一部分，提供了一套通用的模板类和函数，用于实现常用的数据结构和算法。
* **核心组件**：
  * **容器 (Containers)**：存储数据的对象。
  * **迭代器 (Iterators)**：访问容器中元素的通用机制，类似指针。
  * **算法 (Algorithms)**：处理容器中数据的通用函数（如排序、查找）。
  * **仿函数/函数对象 (Functors/Function Objects)**：行为类似函数的对象。
  * **适配器 (Adapters)**：修改容器、迭代器或仿函数的接口。
  * **分配器 (Allocators)**：管理内存。

#### 2. 容器 (Containers)

* **顺序容器 (Sequence Containers)**：元素按线性顺序存储。
  * `std::vector`：动态数组。尾部插入/删除快，随机访问快，中间插入/删除慢。头文件 `<vector>`。
  * `std::list`：双向链表。任意位置插入/删除快，随机访问慢。头文件 `<list>`。
  * `std::deque` (Double-ended Queue)：双端队列。头尾插入/删除快，随机访问较快。头文件 `<deque>`。
  * `std::array` (C++11)：固定大小数组。性能与内置数组相当。头文件 `<array>`。
  * `std::forward_list` (C++11)：单向链表。头文件 `<forward_list>`。
* **关联容器 (Associative Containers)**：元素按键值有序存储（通常基于红黑树）。
  * `std::set`：存储唯一的、有序的键。头文件 `<set>`。
  * `std::multiset`：存储可重复的、有序的键。头文件 `<set>`。
  * `std::map`：存储唯一的键值对 (key-value pair)，键有序。头文件 `<map>`。
  * `std::multimap`：存储可重复的键值对，键有序。头文件 `<map>`。
* **无序关联容器 (Unordered Associative Containers) (C++11)**：元素无序存储（基于哈希表）。查找、插入、删除平均时间复杂度O(1)。
  * `std::unordered_set`, `std::unordered_multiset`：头文件 `<unordered_set>`。
  * `std::unordered_map`, `std::unordered_multimap`：头文件 `<unordered_map>`。
* **容器适配器 (Container Adapters)**：提供特定接口的容器，基于底层顺序容器实现。
  * `std::stack`：栈 (LIFO)。默认基于 `std::deque`。头文件 `<stack>`。
  * `std::queue`：队列 (FIFO)。默认基于 `std::deque`。头文件 `<queue>`。
  * `std::priority_queue`：优先队列。默认基于 `std::vector` 和 `std::less` (大顶堆)。头文件 `<queue>`。

#### 3. 迭代器 (Iterators)

* 提供统一访问容器元素的方式，行为类似指针。
* **主要操作**：
  * `*it`：解引用，获取迭代器指向的元素。
  * `it->member`：访问元素成员 (如果元素是对象)。
  * `++it`, `it++`：移动到下一个元素。
  * `--it`, `it--`：移动到上一个元素 (双向和随机访问迭代器)。
  * `it1 == it2`, `it1 != it2`：比较迭代器。
  * `it + n`, `it - n`, `it1 - it2`：(仅随机访问迭代器)。
* **容器成员函数获取迭代器**：
  * `container.begin()`：指向第一个元素的迭代器。
  * `container.end()`：指向末尾元素之后位置的迭代器（哨兵）。
  * `container.rbegin()`, `container.rend()`：反向迭代器。
  * `container.cbegin()`, `container.cend()`：常量迭代器 (C++11)。
* **迭代器类别**：
  * 输入迭代器 (Input Iterator)：只读，单向，只能前进。
  * 输出迭代器 (Output Iterator)：只写，单向，只能前进。
  * 前向迭代器 (Forward Iterator)：可读写，单向，只能前进。
  * 双向迭代器 (Bidirectional Iterator)：可读写，双向，可前进后退。
  * 随机访问迭代器 (Random Access Iterator)：可读写，可随机访问 (如 `it + n`)。

#### 4. 算法 (Algorithms)

* 定义在 `<algorithm>` 和 `<numeric>` 等头文件中。
* 通常以迭代器范围 `[first, last)` 作为参数。
* **常用算法示例**：
  * 非修改序列操作：`for_each`, `find`, `find_if`, `count`, `count_if`, `equal`, `search`。
  * 修改序列操作：`copy`, `move`, `transform`, `replace`, `fill`, `remove`, `unique`, `reverse`, `rotate`, `random_shuffle` (C++17废弃,用`shuffle`)。
  * 排序和相关操作：`sort`, `stable_sort`, `partial_sort`, `nth_element`, `binary_search`, `lower_bound`, `upper_bound`, `merge`。
  * 数值操作 (`<numeric>`)：`accumulate`, `inner_product`, `partial_sum`, `adjacent_difference`。

## 第十章 输入输出流

### 一、C++流类库简介

#### 1. 文件与流的概念

* **文件 (File)**：物理概念，外部存储介质上信息的集合（如磁盘文件、键盘、显示器）。
* **流 (Stream)**：逻辑概念，C++对I/O设备的抽象。代表程序与设备之间数据流动的通道。
  * 输入流：数据从设备流向程序。
  * 输出流：数据从程序流向设备。

#### 2. C++流类库的特点

* **类型安全 (Type Safe)**：通过重载，编译器可以检查数据类型。
* **易于扩充**：可以为自定义类型重载 `<<` 和 `>>`。
* **统一接口**：对不同设备的I/O操作使用相似的接口。

#### 3. 基本流类 (头文件 `<iostream>`)

* `ios_base`：流类的基类，提供格式化、状态等基础功能（C++标准中 `ios` 的一部分功能移至此）。
* `ios`：继承自 `ios_base`，是输入输出流的基类。
* `istream`：输入流类，支持提取运算符 `>>`。
* `ostream`：输出流类，支持插入运算符 `<<`。
* `iostream`：同时继承 `istream` 和 `ostream`，支持双向I/O。

#### 4. 预定义的流类对象

* `cin` (类型 `istream`)：标准输入流，通常连接到键盘。
* `cout` (类型 `ostream`)：标准输出流，通常连接到显示器。
* `cerr` (类型 `ostream`)：标准错误流，不带缓冲，通常连接到显示器。
* `clog` (类型 `ostream`)：标准日志流，带缓冲，通常连接到显示器。

#### 5. 文件流类 (头文件 `<fstream>`)

* `ifstream`：输入文件流，从 `istream` 派生，用于从磁盘文件读取数据。
* `ofstream`：输出文件流，从 `ostream` 派生，用于向磁盘文件写入数据。
* `fstream`：双向文件流，从 `iostream` 派生，支持对磁盘文件进行读写操作。

### 二、插入与提取运算符重载

* `<<` (插入运算符) 和 `>>` (提取运算符) 已为标准数据类型预定义。
* 为自定义类重载这两个运算符，通常作为友元函数，以便访问类的私有成员，并且使流对象 (`cin`/`cout` 或文件流对象) 作为第一个参数。

    ```cpp
    class Complex { /* ... */
        friend ostream& operator<<(ostream& os, const Complex& c);
        friend istream& operator>>(istream& is, Complex& c);
    };
    ostream& operator<<(ostream& os, const Complex& c) {
        os << c.real << " + " << c.imag << "i";
        return os; // 返回流对象的引用以支持链式操作
    }
    istream& operator>>(istream& is, Complex& c) {
        is >> c.real >> c.imag; // 假设输入格式为 "real imag"
        return is;
    }
    ```

### 三、输入/输出格式控制

#### 1. 格式控制函数 (ios类的成员函数)

* **格式控制标志字 (Format Flags)**：`ios` 类中定义了一系列枚举常量（如 `ios::left`, `ios::right`, `ios::dec`, `ios::hex`, `ios::scientific`, `ios::fixed`, `ios::showpoint` 等）来控制输出格式。
* **`setf(long flags)`**: 设置指定的格式标志位。
* **`setf(long flags, long mask)`**: 先清除 `mask` 指定的位，再设置 `flags`。
* **`unsetf(long flags)`**: 清除指定的格式标志位。
* **`flags()`**: 返回当前的格式标志字。
* **`flags(long newflags)`**: 设置新的格式标志字，覆盖旧的。
* **`width(int w)`**: 设置下一次输出的最小字段宽度 (只对下一次输出有效)。
* **`precision(int p)`**: 设置浮点数的精度。
  * 默认：总有效数字位数。
  * `ios::fixed` 或 `ios::scientific`：小数点后数字位数。
* **`fill(char c)`**: 设置填充字符 (当输出宽度大于数据实际宽度时使用)。

#### 2. 格式控制符 (Manipulators)

* 可以直接在 `<<` 或 `>>` 链式操作中使用。
* **无参控制符 (定义在 `<iostream>`)**:
  * `endl`: 输出换行符并刷新缓冲区。
  * `ends`: 输出空字符 `\0`。
  * `flush`: 刷新输出缓冲区。
  * `ws`: 读取并丢弃前导空白字符 (用于输入流)。
  * `dec`, `hex`, `oct`: 设置整数的基数为十进制、十六进制、八进制。
* **有参控制符 (定义在 `<iomanip>`)**:
  * `setw(int n)`: 设置字段宽度。
  * `setprecision(int n)`: 设置浮点数精度。
  * `setfill(char c)`: 设置填充字符。
  * `setiosflags(long flags)`: 设置格式标志。
  * `resetiosflags(long flags)`: 清除格式标志。
  * `setbase(int base)`: 设置整数基数 (8, 10, 16)。

### 四、磁盘文件的输入与输出

#### 1. 文件的打开与关闭

* **打开文件 (Opening a File)**：
  * 通过文件流对象的构造函数：

        ```cpp
        ofstream outfile("output.txt"); // 默认以 ios::out 打开
        ifstream infile("input.txt");   // 默认以 ios::in 打开
        ```

  * 通过 `open()` 成员函数：

        ```cpp
        ofstream outfile;
        outfile.open("output.txt", ios::out | ios::app); // 追加模式
        ```

  * **文件打开模式 (File Modes - `ios` 枚举常量)**：
    * `ios::in`: 以读方式打开。
    * `ios::out`: 以写方式打开 (如果文件存在则清空，不存在则创建)。
    * `ios::app`: 以追加方式打开 (在文件末尾写入)。
    * `ios::ate`: 打开文件并立即定位到文件末尾。
    * `ios::trunc`: 如果文件存在，则清空其内容。
    * `ios::binary`: 以二进制模式打开文件。
    * 可以用 `|` (位或) 组合多个模式。
* **检查文件是否成功打开**：

    ```cpp
    ifstream infile("data.txt");
    if (!infile) { // 或者 if (infile.fail())
        cerr << "Error opening file!" << endl;
    }
    ```

* **关闭文件 (Closing a File)**：
  * 调用 `close()` 成员函数：`outfile.close();`
  * 文件流对象在析构时会自动关闭文件。显式关闭是个好习惯。

#### 2. 使用插入与提取运算符进行文件读写

* 与 `cin`/`cout` 类似，但操作对象是文件流对象。

    ```cpp
    ofstream ofs("numbers.txt");
    int x = 10; double y = 3.14;
    ofs << x << " " << y << endl; // 写入文件
    ofs.close();

    ifstream ifs("numbers.txt");
    int a; double b;
    ifs >> a >> b; // 从文件读取
    cout << a << " " << b << endl;
    ifs.close();
    ```

* 写入时，数据间需要分隔符 (如空格、换行) 才能被 `>>` 正确读取。

#### 3. 使用成员函数对文件流类对象进行操作

* **字符I/O**:
  * `istream& get(char& ch)`: 读取单个字符。
  * `ostream& put(char ch)`: 写入单个字符。
  * `int get()`: 读取单个字符并返回其ASCII码，或EOF。
* **行I/O**:
  * `istream& getline(char* buffer, streamsize n, char delim = '\n')`: 读取一行到 `buffer`，最多 `n-1` 个字符，或遇到 `delim`。
  * `istream& getline(istream& is, string& str, char delim = '\n')` (全局函数，常用于 `std::string`)
* **二进制I/O (用于非文本数据，如结构体、对象)**:
  * `istream& read(char* buffer, streamsize n)`: 从文件读取 `n` 字节到 `buffer`。
  * `ostream& write(const char* buffer, streamsize n)`: 将 `buffer` 中的 `n` 字节写入文件。

        ```cpp
        struct Record { int id; char name[20]; };
        Record r = {1, "Test"};
        ofstream obfs("record.dat", ios::binary);
        obfs.write(reinterpret_cast<const char*>(&r), sizeof(Record));
        obfs.close();

        Record r_in;
        ifstream ibfs("record.dat", ios::binary);
        ibfs.read(reinterpret_cast<char*>(&r_in), sizeof(Record));
        ibfs.close();
        ```

### 五、文本文件与二进制文件

* **文本文件 (.txt)**：
  * 内容是字符序列，人可读。
  * 数值数据存储为其字符表示 (如整数 `123` 存为字符 `'1'`, `'2'`, `'3'`)。
  * 在某些系统上，行尾符 (`\n`) 可能会被转换 (如 Windows 上的 `\r\n`)。
  * 优点：兼容性好，易于查看和编辑。
  * 缺点：数值数据存储效率低，读写可能涉及格式转换。
* **二进制文件 (.bin, .dat等)**：
  * 内容是字节序列，直接存储数据的内存表示。
  * 优点：数值数据存储效率高，读写速度快，不进行格式转换。
  * 缺点：兼容性差，不易直接阅读。
* 打开文件时用 `ios::binary` 模式指定二进制文件。

### 六、对数据文件的随机访问

* 允许直接跳转到文件中的任意位置进行读写。
* **文件指针 (File Position Pointer)**：每个文件流对象内部维护一个指向文件中当前读/写位置的指针。
* **`seekg(offset, direction)` (seek get)**：移动输入文件指针。
* **`seekp(offset, direction)` (seek put)**：移动输出文件指针。
  * `offset`: 偏移量 (字节数)。
  * `direction`: 起始位置：
    * `ios::beg`: 文件开头。
    * `ios::cur`: 当前位置。
    * `ios::end`: 文件末尾。
* **`tellg()` (tell get)**：返回当前输入文件指针的位置。
* **`tellp()` (tell put)**：返回当前输出文件指针的位置。

    ```cpp
    fstream file("data.bin", ios::in | ios::out | ios::binary);
    file.seekg(2 * sizeof(int), ios::beg); // 定位到第3个int数据处
    int value;
    file.read(reinterpret_cast<char*>(&value), sizeof(int));
    ```

### 七、字符串流 (String Streams)

* 头文件 `<sstream>`。
* 在内存中的 `std::string` 对象上进行格式化的输入输出，行为类似文件流。
* **`istringstream`**: 从字符串读取数据。

    ```cpp
    string data = "10 20.5 Hello";
    istringstream iss(data);
    int i; double d; string s;
    iss >> i >> d >> s; // i=10, d=20.5, s="Hello"
    ```

* **`ostringstream`**: 将数据格式化写入字符串。

    ```cpp
    ostringstream oss;
    int age = 30; string name = "Alice";
    oss << "Name: " << name << ", Age: " << age;
    string result = oss.str(); // result = "Name: Alice, Age: 30"
    ```

* **`stringstream`**: 支持从字符串读和向字符串写。

### 八、其它输入输出函数 (流状态)

* `ios` 类维护流的状态标志位，指示I/O操作的结果。
* **状态标志位**:
  * `ios::goodbit`: 无错误。
  * `ios::eofbit`: 到达文件末尾 (End Of File)。
  * `ios::failbit`: 非致命I/O错误 (如格式错误)，流仍可用但后续操作可能失败。
  * `ios::badbit`: 致命I/O错误 (如磁盘错误)，流损坏。
* **成员函数检查状态**:
  * `good()`: 如果 `goodbit` 置位 (即无错误)，返回 `true`。
  * `eof()`: 如果 `eofbit` 置位，返回 `true`。
  * `fail()`: 如果 `failbit` 或 `badbit` 置位，返回 `true`。
  * `bad()`: 如果 `badbit` 置位，返回 `true`。
  * `operator bool()`: 流对象在布尔上下文中隐式转换为 `!fail()`。
  * `operator!()`: 流对象在布尔上下文中隐式转换为 `fail()`。
* **`clear(iostate state = ios::goodbit)`**: 清除（或设置）流的状态标志。
* **`rdstate()`**: 返回当前流状态字。
