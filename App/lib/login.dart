import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart'
    show
        AssetImage,
        BoxDecoration,
        BoxFit,
        BuildContext,
        Colors,
        Column,
        Container,
        DecorationImage,
        EdgeInsets,
        InputDecoration,
        Key,
        MediaQuery,
        OutlineInputBorder,
        Scaffold,
        Stack,
        State,
        StatefulWidget,
        Text,
        TextField,
        TextStyle,
        Widget;

class MyLogin extends StatefulWidget {
  const MyLogin({Key? key}) : super(key: key);

  @override
  State<MyLogin> createState() => _MyLoginState();
}

class _MyLoginState extends State<MyLogin> {
  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
          image: DecorationImage(
              image: AssetImage('assets/login.png'), fit: BoxFit.cover)),
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: Stack(children: [
          Container(
            padding: const EdgeInsets.only(left: 35, top: 150),
            child: const Text(
              'Welcome\nBack',
              style: TextStyle(color: Colors.white, fontSize: 33),
            ),
          ),
          SingleChildScrollView(
            child: Container(
              padding: EdgeInsets.only(
                  top: MediaQuery.of(context).size.height * 0.5,
                  right: 35,
                  left: 35),
              child: Column(
                children: const [
                  TextField(
                      decoration: InputDecoration(
                          hintText: "Email",
                          border: OutlineInputBorder(
                              borderRadius: BorderRadius.all(Radius.zero)))),
                  SizedBox(
                    height: 30,
                  ),
                  TextField(
                      decoration: InputDecoration(
                          hintText: "Password",
                          border: OutlineInputBorder(
                              borderRadius: BorderRadius.all(Radius.zero))),
                      obscureText: true),
                  SizedBox(
                    height: 40,
                  ),
                  // Row(
                  //  children: [
                  //    Text(
                  //      "Sign In",
                  //      style: TextStyle(
                  //          color: Color(0xff4c505b),
                  //          fontSize: 27, fontWeight: FontWeight.w700),
                  //      ),
                  //    CircleAvatar(
                  //      radius: 30,
                  //      backgroundColor: Color(0xff4c505b),
                  //
                  //    )
                  //    ],
                  // )
                ],
              ),
            ),
          ),
        ]),
      ),
    );
  }
}
