# Issues

## Technical questions

The best way to get help with technical questions is on
[StackOverflow](https://stackoverflow.com/questions/tagged/sigma) using the `[sigma]`
tag. To contact the Microsoft SIGMA team directly, please email
[sigmacrypto@microsoft.com](mailto:sigmacrypto@microsoft.com).

## Bug reports

We appreciate community efforts to find and fix bugs and issues in Microsoft SIGMA.
If you believe you have found a bug or want to report some other issue, please
do so on [GitHub](https://github.com/Microsoft/SIGMA/issues). To help others
determine what the problem may be, we provide a helpful script that collects
relevant system information that you can submit with the bug report (see below).

### System information

To collect system information for an improved bug report, please run
```
make -C tools system_info
```
This will result in a file `system_info.tar.gz` to be generated, which you can
optionally attach with your bug report.

## Critical security issues

For reporting critical security issues, see [SECURITY.md](SECURITY.md).
