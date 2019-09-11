# ImJoy Plugin Repository for UC2 projects Project Template

For the detailed description for the ImJoy software, please have a look at their [website]

0. Folder **`imjoy-plugins`**: contains the ImJoy plugin files (extension 'imjoy.html'). 
1. File **`manifest.imjoy.json`**: specifies how your ImJoy plugins will be shown in the ImJoy plugin import menu
2. Folder **`data`**: contains some sample data to be processed with the plugins

 "name": "UC2-ImJoy Repository",
 "description": "ImJoy plugin repository for UC2 Applications",
 "uri_root": "",
 "version": "0.2.0",
 "plugins": [

## List of available plugins


## Creating url to automatically install your plugins
The ImJoy plugin files can be used to [generate a url](http://imjoy.io/docs/index.html#/development?id=distributing-your-plugin-with-url) which automatically opens ImJoy and installs your plugin with all dependencies. 

This example install the template plugin: [Template Plugin](https://imjoy.io/#/app?plugin=https://raw.githubusercontent.com/oeway/ImJoy-project-template/master/imjoy-plugins/templatePlugin.imjoy.html)

Once the plugin is installed, click `Template Plugin` in the plugin menu and follow the instructions in the dialog to install the Python Plugin Engine. You can then execute the plugin. 

