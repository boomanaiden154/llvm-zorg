"""Implements the command line 'llvmlab' tool."""

import hashlib
import os
import random
import shutil
import sys

import flask
import llvmlab.data
import llvmlab.user
import llvmlab.ci.status
import llvmlab.ui.app

def note(message):
    print >>sys.stderr,"note: %s" % message
def warning(message):
    print >>sys.stderr,"warning: %s" % message

def sorted(items):
    items = list(items)
    items.sort()
    return items
def split_name_and_email(str):
    if (str.count('<') != 1 or
        str.count('>') != 1 or
        not str.endswith('>')):
        raise ValueError,"Don't know how to parse: %r" % (str,)

    lhs,rhs = str[:-1].split("<")
    return lhs.strip(), rhs.strip()

def action_create(name, args):
    """create a llvmlab installation"""

    import llvmlab
    from optparse import OptionParser, OptionGroup
    parser = OptionParser("%%prog %s [options] <path>" % name)
    parser.add_option("-f", "--force", dest="force", action="store_true",
                      help="overwrite existing files")

    group = OptionGroup(parser, "CONFIG OPTIONS")
    group.add_option("", "--admin-login", dest="admin_login",
                      help="administrator login [%default]", default='admin')
    group.add_option("", "--admin-name", dest="admin_name",
                      help="administrator name [%default]",
                      default='Administrator')
    group.add_option("", "--admin-password", dest="admin_password",
                      help="administrator password [%default]", default='admin')
    group.add_option("", "--admin-email", dest="admin_email",
                      help="administrator email [%default]",
                     default='admin@example.com')

    group.add_option("", "--debug-server", dest="debug_server",
                      help="run server in debug mode [%default]",
                     action="store_true", default=False)
    parser.add_option_group(group)

    (opts, args) = parser.parse_args(args)

    if len(args) != 1:
        parser.error("invalid number of arguments")

    basepath, = args
    basepath = os.path.abspath(basepath)
    cfg_path = os.path.join(basepath, 'lab.cfg')
    data_path = os.path.join(basepath, 'lab-data.json')
    status_path = os.path.join(basepath, 'lab-status.json')

    if not os.path.exists(basepath):
        try:
            os.mkdir(basepath)
        except:
            parser.error("unable to create directory: %r" % basepath)
    elif not os.path.isdir(basepath):
        parser.error("%r exists but is not a directory" % basepath)

    if not opts.force:
        if os.path.exists(cfg_path):
            parser.error("%r exists (use --force to override)" % cfg_path)
        if os.path.exists(data_path):
            parser.error("%r exists (use --force to override)" % data_path)
        if os.path.exists(status_path):
            parser.error("%r exists (use --force to override)" % status_path)

    # Construct the config file.
    sample_cfg_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                   "lab.cfg.sample")
    sample_cfg_file = open(sample_cfg_path, "rb")
    sample_cfg_data = sample_cfg_file.read()
    sample_cfg_file.close()

    # Fill in the sample config.
    secret_key = hashlib.sha1(str(random.getrandbits(256))).hexdigest()
    cfg_options = dict(opts.__dict__)
    cfg_options['admin_passhash'] = hashlib.sha256(
        opts.admin_password + secret_key).hexdigest()
    cfg_options['secret_key'] = secret_key
    cfg_options['data_path'] = data_path
    cfg_options['status_path'] = status_path
    cfg_data = sample_cfg_data % cfg_options

    # Write the initial config file.
    cfg_file = open(cfg_path, 'w')
    cfg_file.write(cfg_data)
    cfg_file.close()

    # Construct the initial database and status files.
    data = llvmlab.data.Data(users = [], machines = [])
    status = llvmlab.ci.status.Status({})

    # Construct an app instance, and save the data.
    instance = llvmlab.ui.app.App.create_standalone(data = data,
                                                    status = status,
                                                    config_path = cfg_path)
    instance.save_data()
    instance.save_status()

def action_runserver(name, args):
    """run a llvmlab instance"""

    import llvmlab
    from optparse import OptionParser, OptionGroup
    parser = OptionParser("%%prog %s [options]" % name)
    (opts, args) = parser.parse_args(args)

    if len(args) != 0:
        parser.error("invalid number of arguments")

    instance = llvmlab.ui.app.App.create_standalone()
    instance.run()

def action_import_users(name, args):
    """import users from SVN information"""

    import llvmlab
    import ConfigParser
    from optparse import OptionParser, OptionGroup
    parser = OptionParser("""\
%%prog %s [options] <lab config path> <svn mailer config> <svn htpasswd path>

This command imports user information from the llvm.org SVN information. It will
add any users who are not present in the lab.llvm.org database, and import their
name, email, and SVN login information.\
""" % name)
    (opts, args) = parser.parse_args(args)

    if len(args) != 3:
        parser.error("invalid number of arguments")

    config_path, svn_mailer_path, svn_htpasswd_path = args

    # Load the app object.
    instance = llvmlab.ui.app.App.create_standalone(config_path = config_path)
    data = instance.config.data

    # Load the SVN mailer config.
    parser = ConfigParser.RawConfigParser()
    parser.read(svn_mailer_path)

    # Load the SVN htpasswd file.
    file = open(svn_htpasswd_path)
    svn_htpasswd = {}
    for ln in file:
        if ln.strip():
            user,htpasswd,module = ln.split(":")
            svn_htpasswd[user] = (htpasswd, module)
    file.close()

    # Validate that the authors list and the htpasswd list coincide.
    svn_authors = dict((author, parser.get("authors", author))
                       for author in parser.options("authors"))
    for id in set(svn_authors) - set(svn_htpasswd):
        warning("svn mailer authors contains user without htpasswd: %r " % id)
    for id in set(svn_htpasswd) - set(svn_authors):
        warning("svn contains passwd but no mailer entry: %r " % id)

    # Add user entries for any missing users.
    for id in sorted(set(svn_authors) & set(svn_htpasswd)):
        name,email = split_name_and_email(svn_authors[id])
        htpasswd = svn_htpasswd[id][0]
        passhash = hashlib.sha256(
            htpasswd + instance.config['SECRET_KEY']).hexdigest()

        # Lookup the user entry.
        user = data.users.get(id)

        # Never allow modifying the admin user.
        if user is data.admin_user:
            warning("ignore %r, is the admin user!" % id)
            continue

        # Create the user if missing.
        if user is None:
            # Use the users htpasswd (itself) as the initial password.
            user = data.users[id] = llvmlab.user.User(id, passhash, name,
                                                      email, htpasswd)
            note("added user %r" % id)
            continue

        # Otherwise, update the users info if necessary.
        for kind,new,old in (('name', name, user.name),
                             ('email', email, user.email),
                             ('htpasswd', htpasswd, user.htpasswd)):
            if new != old:
                note("changed %r %s from %r to %r" % (
                        id, kind, old, new))
                setattr(user, kind, new)

    # Save the instance data.
    instance.save_data()

###

commands = dict((name[7:].replace("_","-"), f)
                for name,f in locals().items()
                if name.startswith('action_'))

def usage():
    print >>sys.stderr, "Usage: %s command [options]" % (
        os.path.basename(sys.argv[0]))
    print >>sys.stderr
    print >>sys.stderr, "Available commands:"
    cmds_width = max(map(len, commands))
    for name,func in sorted(commands.items()):
        print >>sys.stderr, "  %-*s - %s" % (cmds_width, name, func.__doc__)
    sys.exit(1)

def main():
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        usage()

    cmd = sys.argv[1]
    commands[cmd](cmd, sys.argv[2:])
