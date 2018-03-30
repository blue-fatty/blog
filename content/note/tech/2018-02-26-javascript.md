---
title: "JavaScript"
date: 2018-02-26T15:14:41+08:00
slug: js
---

This is a note about JavaScript.

## Quick notes

### Ignore SSL issue

**Error:** (node:99469) UnhandledPromiseRejectionWarning: Unhandled promise rejection (rejection id: 4): Error: Hostname/IP doesn't match certificate's altnames: "IP: \<ip\> is not in the cert's list: "

https://github.com/axios/axios/issues/535

``` js
https = require('https')
axios = require('axios')

// At instance level
const instance = axios.create({
  httpsAgent: new https.Agent({  
    rejectUnauthorized: false
  })
});
instance.get('https://something.com/foo');

// At request level
const agent = new https.Agent({  
  rejectUnauthorized: false
});
axios.get('https://something.com/foo', { httpsAgent: agent });
```
<!--more-->
---
