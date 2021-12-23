#include "pj_common.h"
#include "Windows.h"

int main()
{
	auto h = system_clock::now();
	Sleep(100);
	auto i = system_clock::now();
	duration<double> diff = i - h;
	cout << diff.count()*1000 << "ms" << endl;

	system("pause");
	return 0;
}