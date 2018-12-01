import csv
import datetime
import sys
import time

from googleapiclient.discovery import build

# Define terms and home directory if running as script
#QTERMS1 = "flu incubation,flu incubation period,influenza type a,symptoms of the flu,flu symptoms,influenza symptoms,flu contagious,influenza a,a influenza,symptoms of flu,flu duration,influenza incubation,type a influenza,flu treatment,symptoms of influenza,influenza contagious,flu in children,cold or flu,symptoms of bronchitis,flu recovery,tessalon,influenza incubation period,symptoms of pneumonia,tussionex,signs of the flu,flu treatments,remedies for the flu,walking pneumonia,flu test,tussin,upper respiratory,respiratory flu,acute bronchitis,bronchitis,sinus infections,flu relief,painful cough,how long does the flu last,flu cough,sinus,expectorant,strep,strep throat,influenza treatment,flu reports,flu remedy,robitussin,rapid flu,treatment for the flu,chest cold,cough fever,oscillococcinum,flu fever,treat the flu,how to treat the flu,over the counter flu,how long is the flu,flu medicine,flu or cold,normal body,is flu contagious,treat flu,body temperature,reduce fever,flu vs cold,how long is the flu contagious,fever reducer,get over the flu,treating flu,having the flu,treatment for flu,human temperature,dangerous fever,the flu,remedies for flu,influenza a and b,contagious flu,fever flu,flu remedies,how long is flu contagious,cold vs flu,braun thermoscan,fever cough,signs of flu,how long does flu last,normal body temperature,get rid of the flu,i have the flu,taking temperature,flu versus cold,how long flu,flu germs,flu and cold,thermoscan,flu complications,high fever,flu children,the flu virus,how to treat flu,pneumonia,flu headache,ear thermometer,how to get rid of the flu,flu how long,cold and flu,over the counter flu medicine,treating the flu,flu care,how long contagious,fight the flu,reduce a fever,cure the flu,medicine for flu,flu length,cure flu,exposed to flu,low body,early flu symptoms,flu report,incubation period for flu,break a fever,flu contagious period,cold versus flu,what to do if you have the flu,medicine for the flu,flu and fever,flu lasts,incubation period for the flu,do i have the flu,"
#OR_QTERMS = "rsv treatment,flu food,baby with rsv,rsv symptoms,rsv contagious,is rsv contagious,rsv symptoms in adults,what is rsv,symptoms of rsv,how long does rsv last,flu duration,flu recovery,flu lasts,what to eat when you have the flu,influenza a contagious,flu fever,rsv infection,how long is rsv contagious,how to get over the flu,influenza type a,rsv in adults,influenza a,rsv,flu treatment,tamiflu side effects,baby rsv,baby has rsv,get over the flu,signs of rsv,rsv baby,can adults get rsv,coughing remedies,can dogs get the flu from humans,flu treatments,is influenza a contagious,type a influenza,flu gestation,is tamiflu an antibiotic,oseltamivir,influenza a symptoms,when you have the flu,symptoms of rsv in adults,flu contagious,rsv infant,treatment for rsv,respiratory flu,influenza treatment,influenza contagious,flu cough,how to get rid of the flu,getting over the flu,flu recovery time,flu cold,fever flu,recovering from flu,how long contagious"
#NJ_QTERMS = "flu contagious period,oseltamivir,robitussin cf,influenza b,biaxin side effects,tamiflu side effects,tamiflu dosage,tamiflu and pregnancy,type b influenza,how long does the flu last in adults,robitussin cough,influenza type b,influenza b symptoms,type b flu,b flu,type b flu symptoms,symptoms of influenza b,what is influenza b,how long does the flu last?,flu type b,influenza b treatment,man cold,type a flu,clarithromycin,anas barbariae hepatis,flu type,z pack and alcohol,tamiflu drug interactions,tamiflu drug,oscillo,flu cures,tamiflu suspension,c products,flu type b symptoms,stratton lodging,flu treatments,influenza a,tamiflu in children,intestinal flu,is the flu contagious?,oscillococcinum,influenza a incubation period,how to break a fever in adults,flu duration,sinus infection cure,influenza a incubation,influenza treatment guidelines,influenza a treatment,tylenol sinus"

#QTERMS2 = "flu cold,flu?,flu treatment,tamiflu dosage,oscillococcinum,flu food,how to get over the flu,flu duration,flu remedies,rsv,flu cough,symptoms of rsv,how long does rsv last,influenza type a,flu cures,how long contagious flu,flu and pregnancy,getting over the flu,flu in adults,how long is rsv contagious,how to cure the flu,flu lasts,fever flu,contagious flu,how long does the flu last?,flu coughing,flu vomiting,cure for flu,getting rid of the flu,what to eat when you have the flu,cold vs flu symptoms,cures for the flu,flu foods,viral flu,flu medication,best flu medicine,cure the flu,tamiflu wiki,cough flu,oscillococcinum reviews,pregnant with the flu,flu stomach,fever reducers,flu fever,beat the flu,cold versus flu,flu remedy,flu types,how long are you contagious with the stomach flu,intestinal flu,get over the flu,flu cure,when you have the flu,get rid of the flu,best over the counter flu medicine,influenza treatment,how to get rid of the flu,remedies for the flu,type a influenza,treating the flu,baby rsv,phenergan with codeine,flu recovery,treatment for rsv,flu what to do,flu home remedies,cure for the flu,respiratory flu symptoms"
#QTERMS = NJ_QTERMS

HOME_DIR = '/home/fredlu/'


# ------ Insert your API key in the string below. -------
API_KEY = '*****'

SERVER = 'https://www.googleapis.com'
API_VERSION = 'v1beta'
DISCOVERY_URL_SUFFIX = '/discovery/v1/apis/trends/' + API_VERSION + '/rest'
DISCOVERY_URL = SERVER + DISCOVERY_URL_SUFFIX
MAX_QUERIES = 2
TODAY = time.strftime('%Y-%m-%d')


def DateToISOString(datestring):
    """Convert date from (eg) 'Jul 04 2004' to '2004-07-11'.

    Args:
    datestring: A date in the format 'Jul 11 2004', 'Jul 2004', or '2004'

    Returns:
    The same date in the format '2004-11-04'

    Raises:
     ValueError: when date doesn't match one of the three expected formats.
    """

    try:
        new_date = datetime.datetime.strptime(datestring, '%b %d %Y')
    except ValueError:
        try:
            new_date = datetime.datetime.strptime(datestring, '%b %Y')
        except ValueError:
            try:
                new_date = datetime.datetime.strptime(datestring, '%Y')
            except:
                raise ValueError("Date doesn't match any of '%b %d %Y', '%b %Y', '%Y'.")

    return new_date.strftime('%Y-%m-%d')


def GetQueryVolumes(queries, start_date, end_date,
                    geo_level='country', geo='US', frequency='week'):
    """Extract query volumes from Flu Trends API.

    Args:
    queries: A list of all queries to use.
    start_date: Start date for timelines, in form YYYY-MM-DD.
    end_date: End date for timelines, in form YYYY-MM-DD.
    geo: The code for the geography of interest which can be either country
         (eg "US"), region (eg "US-NY") or DMA (eg "501").
    geo_level: The granularity for the geo limitation. Can be "country",
                 "region", or "dma"
    frequency: The time resolution at which to pull queries. One of "day",
                 "week", "month", "year".

    Returns:
    A list of lists (one row per date) that can be output by csv.writer.

    Raises:
    ValueError: when geo_level is not one of "country", "region" or "dma".
    """

    print '------START------'
    print 'geo_level={0}'.format(geo_level)
    print 'geo_id={0}'.format(geo)

    if not API_KEY:
        raise ValueError('API_KEY not set.')

    service = build('trends', API_VERSION,
                    developerKey=API_KEY,
                    discoveryServiceUrl=DISCOVERY_URL)

    dat = {}

    # Note that the API only allows querying 30 queries in one request. In
    # the event that we want to use more queries than that, we need to break
    # our request up into batches of 30.

    # NB: As of Jan. 2017 only 2 queries/second are supported
    batch_intervals = range(0, len(queries), MAX_QUERIES)

    for batch_start in batch_intervals:
        batch_end = min(batch_start + MAX_QUERIES, len(queries))
        query_batch = queries[batch_start:batch_end]

        for attempt in range(5):
            try:
                # Make API query
                if geo_level == 'country':
                    # Country format is ISO-3166-2 (2-letters), e.g. 'US'
                    req = service.getTimelinesForHealth(terms=query_batch,
                                                        time_startDate=start_date,
                                                        time_endDate=end_date,
                                                        timelineResolution=frequency,
                                                        geoRestriction_country=geo)
                elif geo_level == 'dma':
                    # See https://support.google.com/richmedia/answer/2745487
                    req = service.getTimelinesForHealth(terms=query_batch,
                                                        time_startDate=start_date,
                                                        time_endDate=end_date,
                                                        timelineResolution=frequency,
                                                        geoRestriction_dma=geo)
                elif geo_level == 'region':
                    # Region format is ISO-3166-2 (4-letters), e.g. 'US-NY' (see more examples
                    # here: en.wikipedia.org/wiki/ISO_3166-2:US)
                    req = service.getTimelinesForHealth(terms=query_batch,
                                                        time_startDate=start_date,
                                                        time_endDate=end_date,
                                                        timelineResolution=frequency,
                                                        geoRestriction_region=geo)
                else:
                    raise ValueError("geo_level must be one of 'country', 'region' or 'dma'")

                res = req.execute()

                # Sleep for 1.2 seconds so as to avoid hitting rate limiting.
                time.sleep(1.2)

                # Convert the data from the API into a dictionary of the form
                # {(query, date): count, ...}
                res_dict = {(line[u'term'], DateToISOString(point[u'date'])):
                            point[u'value']
                            for line in res[u'lines']
                            for point in line[u'points']}

                # Update the global results dictionary with this batch's results.
                dat.update(res_dict)

                break

            except Exception as e:
                print e
                time.sleep(2.2)

    # Make the list of lists that will be the output of the function
    res = [['date'] + queries]
    for date in sorted(list(set([x[1] for x in dat]))):
        vals = [dat.get((term, date), 0) for term in queries]
        res.append([date] + vals)

    print '------END------'
    return res


def main():
    print '+++ START +++'

    geolevel = sys.argv[1]
    georegion = sys.argv[2]
    freq = sys.argv[3]
    # qterms = sys.argv[3].split(',')
    qterms = QTERMS.split(',')
    # Examples of calling the GetQueryVolumes function for different geo
    # levels and time resolutions.
    # geo can be "US", "US-MA", "506"(boston) respective to geo_level below
    # geo_level can be country, region or dma
    # frequency can be week, day, or month
    getcounts = GetQueryVolumes(qterms,
                                start_date='2004-01-01',
                                end_date=TODAY,
                                geo=georegion,
                                geo_level=geolevel,
                                frequency=freq)

    # Example of writing one of these files out as a CSV file to GTdata.
    csv_out = open(HOME_DIR + 'GTdata.csv', 'wb')
    outwriter = csv.writer(csv_out)
    for row in getcounts:
        outwriter.writerow(row)
    csv_out.close()

    print '+++ END +++'


if __name__ == '__main__':
    main()
