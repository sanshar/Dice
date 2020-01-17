// Question 1 from CAJ
// write a loop to print 1-10

// Question 1.1
// print eg '1:odd; 2:even ...'

#include <iostream>
using namespace std;

int main()
{
	for( int a = 1; a <= 10; a = a + 1)
	{
		if (a%2 ==0)
		{
			cout << a << " : even" << endl;


		}
		else
		{
			cout << a << " : odd" << endl;
		}
		
	}
	return 0;
}
